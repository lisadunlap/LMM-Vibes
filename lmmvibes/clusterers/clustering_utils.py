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
from .clustering_prompts import clustering_systems_prompt, coarse_clustering_systems_prompt
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


def llm_coarse_cluster_with_centers(
    fine_cluster_names: List[str],
    max_coarse_clusters: int = 15,
    verbose: bool = True,
    model: str = "o3",
    cluster_assignment_model: str = "gpt-4.1-mini",
) -> Tuple[Dict[str, str], List[str]]:
    """Use an LLM to create coarse-grained cluster centers from fine-grained clusters."""
    valid_fine_names = [n for n in fine_cluster_names if n != "Outliers"]
    
    if not valid_fine_names:
        logger.warning("No valid fine-grained cluster names found (all outliers)")
        return {}, ["Outliers"]
    
    if len(valid_fine_names) <= max_coarse_clusters:
        # If we already have few enough clusters, return them as-is
        fine_to_coarse = {name: name for name in valid_fine_names}
        return fine_to_coarse, valid_fine_names
    
    fine_cluster_text = "\n".join(valid_fine_names)
    
    system_prompt = f"""You are a machine learning expert specializing in the behavior of large language models. 

I will provide you with a list of fine-grained properties describing model behavior. Your task is to create {max_coarse_clusters} broader property names that capture the high-level themes across these properties.

Instructions:
1. Analyze all the fine-grained properties
2. Identify {max_coarse_clusters} major properties
3. Create clear, descriptive names for each property
4. Each property should be a short sentence or two that captures the essence of that property
5. Output ONLY the property names, one per line
6. Do NOT include numbering, bullets, or other formatting - just the plain property names

Focus on creating properties that are:
- Distinct from each other
- Broad enough to encompass multiple fine-grained properties
- Descriptive and meaningful for understanding model behavior"""
    
    user_prompt = f"Fine-grained properties:\n\n{fine_cluster_text}\n\nGenerate {max_coarse_clusters} coarse-grained property names:"
    
    # Use caching mechanism
    request_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": coarse_clustering_systems_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 1000,
    }

    # Check cache first
    cached_response = _cache.get_completion(request_data)
    if cached_response is not None:
        content = cached_response["choices"][0]["message"]["content"]
        if verbose:
            logger.info("Cache hit for coarse cluster generation!")
    else:
        if verbose:
            logger.info(f"Generating coarse cluster centers using {model}...")
        response = litellm.completion(**request_data, caching=False)
        content = response.choices[0].message.content
        
        # Cache the response
        response_dict = {
            "choices": [{
                "message": {
                    "content": content
                }
            }]
        }
        _cache.set_completion(request_data, response_dict)

    logger.info(content)
    
    # Parse and clean the cluster names, removing any list formatting
    raw_names = [n.strip() for n in content.strip().split("\n") if n.strip()]
    coarse_cluster_names = [_clean_list_item(name) for name in raw_names if _clean_list_item(name)]
    
    if verbose:
        logger.info("Generated concept centres:")
        for i, name in enumerate(coarse_cluster_names):
            logger.info(f"  {i}: {name}")

    # Embeddings for similarity matching
    fine_to_coarse = llm_match(valid_fine_names, coarse_cluster_names, model=cluster_assignment_model)

    return fine_to_coarse, coarse_cluster_names

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
                    logger.warning(f"Error matching fine grained property to coarse grained property: {e}\n\nLabel: {coarse_label}\n\nCoarse names: {coarse_cluster_names}")
   
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


def save_clustered_results(df, base_filename, include_embeddings=True, config=None):
    """Save clustered results in multiple formats and optionally log to wandb."""

    os.makedirs(f"cluster_results/{base_filename}", exist_ok=True)
    
    # Convert problematic columns to strings
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    # Save full results with embeddings
    if include_embeddings:
        full_path = f"cluster_results/{base_filename}/{base_filename}_with_embeddings.parquet"
        df.to_parquet(full_path, compression='snappy')
        logger.info(f"Saved full results to: {full_path}")
    
    # Save lightweight version without embeddings
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    df_light = df.drop(columns=embedding_cols) if embedding_cols else df
    
    light_parquet = f"cluster_results/{base_filename}/{base_filename}_lightweight.parquet"
    light_csv_gz = f"cluster_results/{base_filename}/{base_filename}.csv.gz"
    light_jsonl = f"cluster_results/{base_filename}/{base_filename}.jsonl"
    summary_table = f"cluster_results/{base_filename}/{base_filename}_summary_table.jsonl"
    
    df_light.to_parquet(light_parquet, compression='snappy')
    df_light.to_csv(light_csv_gz, index=False, compression='gzip')
    df_light.to_json(light_jsonl, lines=True, orient="records")

    summary_table = create_summary_table(df_light, config)
    summary_table.to_json(summary_table, lines=True, orient="records")
    
    logger.info(f"Saved lightweight results to: {light_parquet}, {light_csv_gz}, {light_jsonl}")
    
    # Print file sizes
    if include_embeddings:
        full_size = os.path.getsize(full_path) / (1024**2)
        logger.info(f"  Full dataset: {full_size:.1f} MB")
    
    light_size = os.path.getsize(light_parquet) / (1024**2)
    csv_gz_size = os.path.getsize(light_csv_gz) / (1024**2)
    logger.info(f"  Lightweight: {light_size:.1f} MB (parquet), {csv_gz_size:.1f} MB (csv.gz)")

    # Log to wandb if enabled
    if config and config.use_wandb:
        log_results_to_wandb(df_light, light_csv_gz, base_filename, config)


def log_results_to_wandb(df_light, csv_path, base_filename, config):
    """Log clustering results to wandb."""
    
    if not wandb.run:
        logger.warning("⚠️ wandb not initialized, skipping logging")
        return
    
    logger.info("📊 Logging results to wandb...")
    
    try:
        # Log the lightweight CSV file
        artifact = wandb.Artifact(
            name=f"{base_filename}_clustered_data",
            type="clustered_dataset",
            description=f"Clustered dataset without embeddings - {base_filename}"
        )
        artifact.add_file(csv_path)
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
        
    except Exception as e:
        logger.error(f"❌ Failed to log to wandb: {e}")


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
    # cols = ['model_1_name', 'model_2_name', 'winner',
    #     'question_id', 'prompt', 'model_1_response',
    #     'model_2_response', 'differences', 'model',
    #     'property_description', 'category', 'evidence', 'type', 'reason',
    #     'impact', 'contains_errors', 'unexpected_behavior'
    #     ]
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
            "coarse_label": df_label.property_description_coarse_cluster_label.value_counts().idxmax(),
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results