"""
clustering_utils.py

Utility functions shared across clustering modules, particularly for 
LLM‐based summarisation and embedding handling.

These were originally defined in hierarchical_clustering.py but were 
extracted to make them re-usable across different scripts and to keep the
main algorithm file focussed on clustering logic.
"""

from __future__ import annotations

import concurrent.futures
import random
import time
import logging
from typing import List, Tuple, Dict
import os
import pickle
import pandas as pd
import wandb

import numpy as np
import litellm  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from .clustering_prompts import clustering_systems_prompt, coarse_clustering_systems_prompt, deduplication_clustering_systems_prompt
from ..core.caching import LMDBCache

# Global cache instance
def _get_cache_config():
    """Get cache configuration from environment variables."""
    import os
    
    # Get cache directory from environment or use default
    cache_dir = os.environ.get("LITELLM_CACHE_DIR_CLUSTERING", ".cache/lmmvibes/clustering")
    
    # Get cache size from environment or use default
    cache_size = os.environ.get("LITELLM_CACHE_SIZE", "10GB")
    
    return cache_dir, cache_size

_cache_dir, _cache_size = _get_cache_config()
_cache = LMDBCache(cache_dir=_cache_dir, max_size=_cache_size)

# Module-level logger
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Ensure deterministic behaviour in sampling helpers and downstream clustering
# -----------------------------------------------------------------------------
# A single global seed keeps `random.sample` calls reproducible across runs.
random.seed(42)

# -----------------------------------------------------------------------------
# OpenAI embeddings helpers
# -----------------------------------------------------------------------------

def _get_openai_embeddings_batch(batch: List[str], retries: int = 3, sleep_time: float = 2.0):
    """Fetch embeddings for one batch with simple exponential back-off."""
    
    # Check cache first for each text in batch
    cached_embeddings = []
    texts_to_embed = []
    indices_to_embed = []
    
    for i, text in enumerate(batch):
        cached_emb = _cache.get_embedding(text)
        if cached_emb is not None:
            cached_embeddings.append((i, cached_emb))
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)
    
    # If all embeddings were cached, return them in order
    if not texts_to_embed:
        # Sort by original index and return only embeddings
        return [emb for _, emb in sorted(cached_embeddings, key=lambda x: x[0])]
    
    # Get embeddings for texts not in cache
    for attempt in range(retries):
        try:
            resp = litellm.embedding(
                model="text-embedding-3-large",
                input=texts_to_embed,
                caching=False,  # Disable litellm caching since we're using our own
            )
            new_embeddings = [item["embedding"] for item in resp["data"]]
            
            # Cache the new embeddings
            for text, embedding in zip(texts_to_embed, new_embeddings):
                _cache.set_embedding(text, embedding)
            
            # Combine cached and new embeddings
            all_embeddings = [None] * len(batch)
            for i, emb in cached_embeddings:
                all_embeddings[i] = emb
            for i, emb in zip(indices_to_embed, new_embeddings):
                all_embeddings[i] = emb
                
            return all_embeddings
            
        except Exception as exc:
            if attempt == retries - 1:
                raise
            logger.warning(f"[retry {attempt + 1}/{retries}] {exc}. Sleeping {sleep_time}s.")
            time.sleep(sleep_time)


def _get_openai_embeddings(texts: List[str], *, batch_size: int = 100, max_workers: int = 10) -> List[List[float]]:
    """Get embeddings for *texts* from the OpenAI API whilst preserving order."""

    if not texts:
        return []

    embeddings: List[List[float] | None] = [None] * len(texts)
    batches = [(start, texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_span = {
            executor.submit(_get_openai_embeddings_batch, batch_texts): (start, len(batch_texts))
            for start, batch_texts in batches
        }
        for fut in concurrent.futures.as_completed(future_to_span):
            start, length = future_to_span[fut]
            batch_embeddings = fut.result()
            embeddings[start : start + length] = batch_embeddings

    if any(e is None for e in embeddings):
        raise RuntimeError("Some embeddings are missing – check logs for errors.")

    # mypy: we just checked there are no Nones
    return embeddings  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Generic embedding helper
# -----------------------------------------------------------------------------

def _get_embeddings(texts: List[str], embedding_model: str, verbose: bool = False) -> List[List[float]]:
    """Return embeddings for *texts* using either OpenAI or a SentenceTransformer."""

    if embedding_model == "openai":
        return _get_openai_embeddings(texts)

    if verbose:
        logger.info(f"Computing embeddings with {embedding_model}…")

    model = SentenceTransformer(embedding_model)
    return model.encode(texts, show_progress_bar=verbose).tolist()

# -----------------------------------------------------------------------------
# LLM-based cluster summarisation helpers
# -----------------------------------------------------------------------------

def _get_llm_cluster_summary(
    values: List[str],
    model: str,
    column_name: str,
    cluster_type: str,
    sample_size: int = 50,
) -> str:
    """Generate a short human readable summary for a cluster via LLM.

    This is a lightly refactored version of _get_llm_cluster_summary from
    *hierarchical_clustering.py* with the leading underscore removed to
    signal its intended public use from the utils module.
    """

    sampled_vals = values if len(values) <= sample_size else random.sample(values, sample_size)
    values_text = "\n".join(map(str, sampled_vals))

    # Build request data for caching
    request_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": clustering_systems_prompt},
            {"role": "user", "content": values_text}
        ],
        "max_completion_tokens": 100,
    }

    # Check cache first
    cached_response = _cache.get_completion(request_data)
    if cached_response is not None:
        content = cached_response["choices"][0]["message"]["content"].strip()
    else:
        response = litellm.completion(
            **request_data,
            caching=False,  # Disable litellm caching since we're using our own
        )
        content = response.choices[0].message.content.strip()
        
        # Cache the response
        response_dict = {
            "choices": [{
                "message": {
                    "content": content
                }
            }]
        }
        _cache.set_completion(request_data, response_dict)

    if content.startswith(("'", '"')):
        content = content[1:]
    if content.endswith(("'", '"')):
        content = content[:-1]
    return content


def _clean_list_item(text: str) -> str:
    """
    Clean up numbered or bulleted list items to extract just the content.
    
    Handles formats like:
    - "1. Item name" -> "Item name"
    - "• Item name" -> "Item name" 
    - "- Item name" -> "Item name"
    - "* Item name" -> "Item name"
    - "a) Item name" -> "Item name"
    - "i. Item name" -> "Item name"
    """
    import re
    
    # Remove common list prefixes
    patterns = [
        r'^\s*\d+\.\s*',           # "1. ", "10. ", etc.
        r'^\s*\d+\)\s*',           # "1) ", "10) ", etc.
        r'^\s*[a-zA-Z]\.\s*',      # "a. ", "A. ", etc.
        r'^\s*[a-zA-Z]\)\s*',      # "a) ", "A) ", etc.
        r'^\s*[ivxlc]+\.\s*',      # Roman numerals "i. ", "iv. ", etc.
        r'^\s*[IVXLC]+\.\s*',      # "I. ", "IV. ", etc.
        r'^\s*[•·◦▪▫‣⁃]\s*',       # Bullet characters
        r'^\s*[-*+]\s*',           # Dash, asterisk, plus bullets
        r'^\s*[→⟶]\s*',            # Arrow bullets
    ]
    
    cleaned = text.strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    return cleaned.strip()


def generate_coarse_labels(
    fine_cluster_names: List[str],
    max_coarse_clusters: int,
    *,
    systems_prompt: str = deduplication_clustering_systems_prompt,
    model: str = "gpt-4.1",
    verbose: bool = True,
) -> List[str]:
    """Return a cleaned list of coarse-grained labels created by an LLM.

    This function is *pure* w.r.t. its inputs: it never mutates global
    state other than consulting / writing to the LMDB cache.
    """
    valid_fine_names = [n for n in fine_cluster_names if n != "Outliers"]
    if max_coarse_clusters and len(valid_fine_names) > max_coarse_clusters:
        systems_prompt = systems_prompt.format(max_coarse_clusters=max_coarse_clusters)

    if not valid_fine_names:
        return ["Outliers"]

    # If the list is already small, just return it unchanged.
    if max_coarse_clusters and len(valid_fine_names) <= max_coarse_clusters:
        return valid_fine_names

    user_prompt = f"Fine-grained properties:\n\n" + "\n".join(valid_fine_names)

    request_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": systems_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 2000,
    }

    # Try cache first
    cached = _cache.get_completion(request_data)
    if cached is not None:
        content = cached["choices"][0]["message"]["content"]
        if verbose:
            logger.info("Cache hit for coarse label generation!")
    else:
        if verbose:
            logger.info(f"Generating {max_coarse_clusters} coarse labels using {model}…")
        resp = litellm.completion(**request_data, caching=False)
        content = resp.choices[0].message.content
        # Store in cache
        _cache.set_completion(request_data, {
            "choices": [{"message": {"content": content}}]
        })

    # Clean and split response into individual labels
    raw_names = [line.strip() for line in content.split("\n") if line.strip()]
    coarse_labels = [_clean_list_item(name) for name in raw_names if _clean_list_item(name)]

    if verbose:
        logger.info("Generated coarse labels:")
        for i, lbl in enumerate(coarse_labels):
            logger.info(f"  {i}: {lbl}")

    return coarse_labels


def assign_fine_to_coarse(
    fine_cluster_names: List[str],
    coarse_cluster_names: List[str],
    *,
    model: str = "gpt-4.1-mini",
    strategy: str = "llm",
    verbose: bool = True,
) -> Dict[str, str]:
    """Assign each fine cluster name to one of the coarse cluster names.

    Parameters
    ----------
    strategy : "llm" | "embedding"
        • "llm" – use chat-based matching (thread-pooled, relies on litellm).
        • "embedding" – cosine-similarity in embedding space (fast, no chat calls).
    """
    if strategy == "embedding":
        return embedding_match(fine_cluster_names, coarse_cluster_names)
    elif strategy == "llm":
        return llm_match(fine_cluster_names, coarse_cluster_names, model=model)
    else:
        raise ValueError(f"Unknown assignment strategy: {strategy}")


def llm_coarse_cluster_with_centers(
    fine_cluster_names: List[str],
    max_coarse_clusters: int,
    verbose: bool = True,
    model: str = "gpt-4.1",
    cluster_assignment_model: str = "gpt-4.1-mini",
    systems_prompt: str = deduplication_clustering_systems_prompt,
) -> Tuple[Dict[str, str], List[str]]:
    """High-level convenience wrapper that returns both mapping and centres."""
    valid_fine_names = [n for n in fine_cluster_names if n != "Outliers"]
    if not valid_fine_names:
        return {}, ["Outliers"]

    coarse_labels = generate_coarse_labels(
        valid_fine_names,
        max_coarse_clusters=max_coarse_clusters,
        systems_prompt=systems_prompt,
        model=model,
        verbose=verbose,
    )

    fine_to_coarse = assign_fine_to_coarse(
        valid_fine_names,
        coarse_labels,
        model=cluster_assignment_model,
        strategy="llm",
        verbose=verbose,
    )

    return fine_to_coarse, coarse_labels

def embedding_match(fine_cluster_names, coarse_cluster_names):
    """Match fine-grained cluster names to coarse-grained cluster names using embeddings."""
    fine_emb = _get_embeddings(fine_cluster_names, "openai", verbose=False)
    coarse_emb = _get_embeddings(coarse_cluster_names, "openai", verbose=False)
    fine_emb = np.array(fine_emb) / np.linalg.norm(fine_emb, axis=1, keepdims=True)
    coarse_emb = np.array(coarse_emb) / np.linalg.norm(coarse_emb, axis=1, keepdims=True)
    sim = fine_emb @ coarse_emb.T
    fine_to_coarse = np.argmax(sim, axis=1)
    # turn into dictionary of {fine_name: coarse_name}
    return {fine_cluster_names[i]: coarse_cluster_names[j] for i, j in enumerate(fine_to_coarse)}

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def match_label_names(label_name, label_options):
    """See if label_name is in label_options, not taking into account capitalization or whitespace or punctuation. Return original option if found, otherwise return None"""
    if "outliers" in label_name.lower():
        return "Outliers"
    label_name_clean = label_name.lower().strip().replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("'", "").replace("\"", "").replace("`", "").replace("~", "").replace("*", "").replace("+", "").replace("-", "").replace("_", "").replace("=", "").replace("|", "").replace("\\", "").replace("/", "").replace("<", "").replace(">", "").replace(" ", "")
    for option in label_options:
        option_clean = option.lower().strip().replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("'", "").replace("\"", "").replace("`", "").replace("~", "").replace("*", "").replace("+", "").replace("-", "").replace("_", "").replace("=", "").replace("|", "").replace("\\", "").replace("/", "").replace("<", "").replace(">", "").replace(" ", "")
        if label_name_clean in option_clean:
            return option
    return None

def llm_match(fine_cluster_names, coarse_cluster_names, max_workers=10, model="gpt-4.1-mini"):
    """Match fine-grained cluster names to coarse-grained cluster names using an LLM with threading."""
    coarse_names_text = "\n".join(coarse_cluster_names)
    fine_to_coarse = {}
    lock = threading.Lock()
    
    system_prompt = "You are a machine learning expert specializing in the bhavior of large langauge models. Given the following coarse grained properties of model behavior, match the given fine grained property to the coarse grained property that it most closely resembles. Respond with the name of the coarse grained property that the fine grained property most resembles. If is okay if the match is not perfect, just respond with the property that is most similar. If the fine grained property has absolutely no relation to any of the coarse grained properties, respond with 'Outliers'. Do NOT include anything but the name of the coarse grained property in your response."
    
    def process_single_fine_name(fine_name):
        """Process a single fine-grained cluster name."""
        retries = 3
        for attempt in range(retries):
            try:
                user_prompt = f"Coarse grained properties:\n\n{coarse_names_text}\n\nFine grained property: {fine_name}\n\nClosest coarse grained property:"
                
                response = litellm.completion(model=model, 
                                            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 
                                            caching=retries == 3)
                coarse_label = response.choices[0].message.content.strip()
                coarse_label = match_label_names(coarse_label, coarse_cluster_names)

                if coarse_label == "Outliers" or coarse_label in coarse_cluster_names:
                    user_prompt = f"Coarse grained properties:\n\n{coarse_names_text}\n\nFine grained property: {coarse_label}\n\nClosest coarse grained property:"
                    response = litellm.completion(model=model, 
                                                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 
                                                caching=True)
                    coarse_label = response.choices[0].message.content.strip()
                    coarse_label = match_label_names(coarse_label, coarse_cluster_names)
                
                assert (coarse_label in coarse_cluster_names) or (coarse_label == "Outliers"), f"Fine grained property {fine_name} does not match any coarse grained property"
                
                # Thread-safe assignment
                with lock:
                    fine_to_coarse[fine_name] = coarse_label
                return fine_name, coarse_label
                
            except Exception as e:
                if attempt == retries - 1:
                    with lock:
                        fine_to_coarse[fine_name] = "Outliers"
                    logger.warning(f"Failed to match fine grained property {fine_name} after {retries} attempts, setting to 'Outliers'")
                    return fine_name, "Outliers"
                else:
                    logger.warning(f"Error matching fine grained property to coarse grained property: {e}\n\nLabel: {coarse_label}")
   
        return fine_name, "Outliers"
    
    # Use ThreadPoolExecutor to process fine cluster names in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_name = {executor.submit(process_single_fine_name, fine_name): fine_name 
                         for fine_name in fine_cluster_names}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_name), total=len(fine_cluster_names), desc="Matching fine to coarse clusters"):
            fine_name = future_to_name[future]
            try:
                result = future.result()
                # Result is already stored in fine_to_coarse by the worker function
            except Exception as e:
                logger.error(f"Exception occurred while processing {fine_name}: {e}")
                with lock:
                    fine_to_coarse[fine_name] = "Outliers"
    
    return fine_to_coarse

def _setup_embeddings(texts, embedding_model, verbose=False):
    """Setup embeddings based on model type. Uses LMDB-based caching."""
    if embedding_model == "openai":
        if verbose:
            logger.info("Using OpenAI embeddings (with LMDB caching)...")
        embeddings = _get_openai_embeddings(texts)
        embeddings = np.array(embeddings)
        # Normalize embeddings
        embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
        return embeddings, None
    else:
        if verbose:
            logger.info(f"Using sentence transformer: {embedding_model}")
        model = SentenceTransformer(embedding_model)
        return None, model


# -------------------------------------------------------------------------
# Legacy Litellm-based helpers (kept for reference – not used by pipeline)
# -------------------------------------------------------------------------
def _get_openai_embeddings_batch_litellm(batch, retries=3, sleep_time=2.0):
    """Fetch embeddings for one batch with retry logic (uses LiteLLM cache)."""
    for attempt in range(retries):
        try:
            resp = litellm.embedding(
                model="text-embedding-3-large",
                input=batch,
                caching=True
            )
            embeddings = [item["embedding"] for item in resp["data"]]
            return embeddings
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"[retry {attempt + 1}/{retries}] {e}. Sleeping {sleep_time}s.")
            time.sleep(sleep_time)


# NOTE: renamed to avoid overriding the LMDB-cached version defined earlier
def _get_openai_embeddings_litellm(texts, batch_size=100, max_workers=10):
    """Get embeddings using OpenAI API (LiteLLM cache)."""

    if not texts:
        return []

    # Pre-allocate output list to preserve order
    embeddings = [None] * len(texts)

    # Prepare batches with their position in the output list
    batches = [
        (start, texts[start:start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_span = {
            executor.submit(_get_openai_embeddings_batch_litellm, batch_texts): (start, len(batch_texts))
            for start, batch_texts in batches
        }

        for fut in concurrent.futures.as_completed(future_to_span):
            start, length = future_to_span[fut]
            batch_embeddings = fut.result()   # exceptions propagate – fail fast
            embeddings[start:start + length] = batch_embeddings

    # Final sanity-check
    if any(e is None for e in embeddings):
        raise RuntimeError("Some embeddings are missing – check logs for errors.")

    return embeddings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_clustered_results(parquet_path):
    """Load previously clustered results from parquet file."""
    df = pd.read_parquet(parquet_path)
    
    logger.info(f"Loaded {len(df)} rows from {parquet_path}")
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower() or 'topic' in col.lower()]
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    
    if cluster_cols:
        logger.info(f"Cluster columns: {cluster_cols}")
    if embedding_cols:
        logger.info(f"Embedding columns: {embedding_cols}")
    
    return df


def save_clustered_results(df, base_filename, include_embeddings=True, config=None, output_dir=None):
    """Save clustered results in multiple formats and optionally log to wandb.
    
    Args:
        df: DataFrame with clustered results
        base_filename: Base name for output files
        include_embeddings: Whether to include embeddings in output
        config: ClusterConfig object for wandb logging
        output_dir: Output directory (if None, uses cluster_results/{base_filename})
    """

    # Determine output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_dir = output_dir
    else:
        os.makedirs(f"cluster_results/{base_filename}", exist_ok=True)
        save_dir = f"cluster_results/{base_filename}"
    
    # Convert problematic columns to strings for JSON serialization
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    # 1. Save clustered results as JSON (preserves all data structures)
    df.to_json(f"{save_dir}/clustered_results.jsonl", orient='records', lines=True)
    logger.info(f"Saved clustered results (JSON): {save_dir}/clustered_results.jsonl")
    
    # 2. Save embeddings separately if they exist
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    if embedding_cols and include_embeddings:
        # Create embeddings-only DataFrame
        embedding_df = df[embedding_cols].copy()
        
        # Add key columns for reference
        key_cols = ['property_description', 'question_id', 'model', 'property_description_fine_cluster_label']
        for col in key_cols:
            if col in df.columns:
                embedding_df[col] = df[col]
        
        # Save embeddings as parquet (more efficient for large arrays)
        embeddings_path = os.path.join(save_dir, "embeddings.parquet")
        embedding_df.to_parquet(embeddings_path, compression='snappy')
        logger.info(f"Saved embeddings: {embeddings_path}")
        
        # Also save as JSON for compatibility
        embedding_df.to_json(f"{save_dir}/embeddings.jsonl", orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved embeddings (JSON): {save_dir}/embeddings.jsonl")
    
    # 3. Save lightweight version without embeddings
    df_light = df.drop(columns=embedding_cols) if embedding_cols else df
    
    # Save lightweight as json
    df_light.to_json(f"{save_dir}/clustered_results_lightweight.jsonl", orient='records', lines=True)
    logger.info(f"Saved lightweight results (JSON): {save_dir}/clustered_results_lightweight.jsonl")
    
    # 4. Create and save summary table
    summary_table = create_summary_table(df_light, config)
    summary_table.to_json(f"{save_dir}/summary_table.jsonl", orient='records', lines=True)
    logger.info(f"Saved summary table: {save_dir}/summary_table.jsonl")

    # 6. Log to wandb if enabled
    if config and config.use_wandb:
        log_results_to_wandb(df_light, f"{save_dir}/clustered_results_lightweight.jsonl", base_filename, config)
        logger.info(f"Logged results to wandb")
    
    return {
        'clustered_json': f"{save_dir}/clustered_results.jsonl",
        'embeddings_parquet': f"{save_dir}/embeddings.parquet" if embedding_cols and include_embeddings else None,
        'summary_table': f"{save_dir}/summary_table.jsonl"
    }


def log_results_to_wandb(df_light, light_json_path, base_filename, config):
    """Log clustering results to wandb."""
    
    if not wandb.run:
        logger.warning("⚠️ wandb not initialized, skipping logging")
        return
    
    logger.info("📊 Logging results to wandb...")
    
    # Log the lightweight CSV file
    artifact = wandb.Artifact(
        name=f"{base_filename}_clustered_data",
        type="clustered_dataset",
        description=f"Clustered dataset without embeddings - {base_filename}"
    )
    artifact.add_file(light_json_path)
    wandb.log_artifact(artifact)
    
    # Log the actual clustering results as a table
    # Find the original column that was clustered
    original_col = None
    for col in df_light.columns:
        if not any(suffix in col for suffix in ['_cluster', '_embedding']):
            # This is likely the original column
            original_col = col
            break
    
    if original_col:
        # Create a table with the key clustering results
        cluster_cols = [col for col in df_light.columns if 'cluster' in col.lower()]
        table_cols = [original_col] + cluster_cols
        
        # Sample the data if it's too large (wandb has limits)
        sample_size = min(100, len(df_light))
        if len(df_light) > sample_size:
            df_sample = df_light[table_cols].sample(n=sample_size, random_state=42)
            logger.info(f"📋 Logging sample of {sample_size} rows (out of {len(df_light)} total)")
        else:
            df_sample = df_light[table_cols]
            logger.info(f"📋 Logging all {len(df_sample)} rows")
        
        # Convert to string to handle any non-serializable data
        df_sample_str = df_sample.astype(str)
        wandb.log({f"{base_filename}_clustering_results": wandb.Table(dataframe=df_sample_str)})
    
    # Calculate clustering metrics
    cluster_cols = [col for col in df_light.columns if 'cluster_id' in col.lower()]
    metrics = {"clustering_dataset_size": len(df_light)}
    
    for col in cluster_cols:
        cluster_ids = df_light[col].values
        n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
        n_outliers = list(cluster_ids).count(-1)
        
        level = "fine" if "fine" in col else "coarse" if "coarse" in col else "main"
        metrics[f"clustering_{level}_clusters"] = n_clusters
        metrics[f"clustering_{level}_outliers"] = n_outliers
        metrics[f"clustering_{level}_outlier_rate"] = n_outliers / len(cluster_ids) if len(cluster_ids) > 0 else 0
        
        # Calculate cluster size distribution
        cluster_sizes = [list(cluster_ids).count(cid) for cid in set(cluster_ids) if cid != -1]
        if cluster_sizes:
            metrics[f"clustering_{level}_avg_cluster_size"] = np.mean(cluster_sizes)
            metrics[f"clustering_{level}_min_cluster_size"] = min(cluster_sizes)
            metrics[f"clustering_{level}_max_cluster_size"] = max(cluster_sizes)
    
    # Log clustering configuration
    config_dict = {
        "clustering_min_cluster_size": config.min_cluster_size,
        "clustering_embedding_model": config.embedding_model,
        "clustering_hierarchical": config.hierarchical,
        "clustering_assign_outliers": config.assign_outliers,
        "clustering_disable_dim_reduction": config.disable_dim_reduction,
        "clustering_min_samples": config.min_samples,
        "clustering_cluster_selection_epsilon": config.cluster_selection_epsilon
    }
    
    # Log all metrics as summary metrics (not regular metrics)
    # Note: This function doesn't have access to WandbMixin, so we'll log directly to wandb.run.summary
    all_metrics = {**metrics, **config_dict}
    for key, value in all_metrics.items():
        wandb.run.summary[key] = value
    
    logger.info(f"✅ Logged clustering results to wandb")
    logger.info(f"   - Dataset artifact: {base_filename}_clustered_data")
    logger.info(f"   - Clustering results table: {base_filename}_clustering_results")
    logger.info(f"   - Summary metrics: {list(all_metrics.keys())}")


def initialize_wandb(config, method_name, input_file):
    """Initialize wandb logging if enabled."""
    if not config.use_wandb:
        return
    
    logger.info("🔧 Initializing wandb...")
    
    # Create run name if not provided
    run_name = config.wandb_run_name
    if not run_name:
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        run_name = f"{input_basename}_{method_name}_clustering"
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project or "hierarchical_clustering",
        entity=config.wandb_entity,
        name=run_name,
        config={
            "method": method_name,
            "input_file": input_file,
            "min_cluster_size": config.min_cluster_size,
            "embedding_model": config.embedding_model,
            "hierarchical": config.hierarchical,
            "assign_outliers": config.assign_outliers,
            "disable_dim_reduction": config.disable_dim_reduction,
            "min_samples": config.min_samples,
            "cluster_selection_epsilon": config.cluster_selection_epsilon
        }
    )
    
    logger.info(f"✅ Initialized wandb run: {run_name}")


def load_precomputed_embeddings(embeddings_path, verbose=True):
    """Load precomputed embeddings from various file formats."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    if verbose:
        logger.info(f"Loading precomputed embeddings from {embeddings_path}...")
    
    if embeddings_path.endswith('.pkl'):
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                # Check if it's a cache file with 'embeddings' key
                if 'embeddings' in data:
                    embeddings = data['embeddings']
                    if verbose:
                        logger.info(f"Loaded {len(embeddings)} embeddings from cache file")
                else:
                    # Assume it's a direct mapping of values to embeddings
                    embeddings = data
                    if verbose:
                        logger.info(f"Loaded {len(embeddings)} embeddings from mapping file")
            else:
                # Assume it's a direct array/list of embeddings
                embeddings = data
                if verbose:
                    logger.info(f"Loaded {len(embeddings)} embeddings from array file")
    
    elif embeddings_path.endswith('.npy'):
        embeddings = np.load(embeddings_path)
        if verbose:
            logger.info(f"Loaded {len(embeddings)} embeddings from numpy file")
    
    elif embeddings_path.endswith('.parquet'):
        # Load from parquet file with embedding column
        if verbose:
            logger.info("Loading parquet file...")
        df = pd.read_parquet(embeddings_path)
        
        embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
        if not embedding_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}")
        
        embedding_col = embedding_cols[0]  # Use first embedding column
        
        # Find the column that was clustered (should be the base name of embedding column)
        base_col = embedding_col.replace('_embedding', '')
        if base_col not in df.columns:
            # Try to find any text column that might be the source
            text_cols = [col for col in df.columns if col not in embedding_cols and 
                        df[col].dtype == 'object']
            if text_cols:
                base_col = text_cols[0]
                if verbose:
                    logger.info(f"Using column '{base_col}' as source column")
            else:
                raise ValueError(f"Cannot find source text column in {embeddings_path}")
        
        if verbose:
            logger.info(f"Creating value-to-embedding mapping from column '{base_col}'...")
        
        # Create mapping from values to embedding
        embeddings = {}
        for _, row in df.iterrows():
            value = str(row[base_col])
            embedding = row[embedding_col]
            embeddings[value] = embedding
        
        if verbose:
            logger.info(f"Loaded {len(embeddings)} embeddings from parquet file (column: {embedding_col})")
    
    else:
        raise ValueError(f"Unsupported file format: {embeddings_path}. Supported: .pkl, .npy, .parquet")
    
    return embeddings

def create_summary_table(df, config=None, **kwargs):
    labels = df.property_description_fine_cluster_label.value_counts()
    cols = [
        'property_description',
        ]
    existing_cols = [c for c in cols if c in df.columns]
    results = []
    for label in labels.index:
        df_label = df[df.property_description_fine_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "fine_label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        if "property_description_coarse_cluster_label" in df_label.columns:
            res["coarse_label"] = df_label.property_description_coarse_cluster_label.value_counts().idxmax()
        results.append(res)

    results = pd.DataFrame(results)
    return results