"""Dataset loading utilities for LMM-Vibes.

This module migrates the helpers that previously lived in *data_loader.py*
into the package namespace so they can be imported as

    from lmmvibes.datasets import load_arena_data, load_webdev_data, load_data

The logic is copied verbatim (with minor import-path tweaks) to keep
back-compatibility with existing scripts.
"""

from __future__ import annotations

from typing import Tuple, Callable, Any, Dict

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Helpers to extract conversation content
# ---------------------------------------------------------------------------

def _extract_content_arena(conversation):
    """Extract content for standard arena data."""
    return conversation[0]["content"], conversation[1]["content"]

selected_keys = [
    "code",
    "commentary",
    "description",
    "file_path",
    "has_additional_dependencies",
    "additional_dependencies",
    "install_dependencies_command",
    "port",
    "template",
    "title",
]

def _extract_content_webdev(conversation):
    """Extract content for webdev arena data."""
    formatted_object = ""
    for key in selected_keys:
        formatted_object += f"## {key}\n{conversation[1]['object'][key]}\n\n"
    formatted_response = (
        f"## Text response\n{conversation[1]['content'][0]['text']}\n\n{formatted_object}## Logs\n{conversation[1]['result']}"
    )
    return conversation[0]["content"][0]["text"], formatted_response

# ---------------------------------------------------------------------------
# Arena loaders
# ---------------------------------------------------------------------------

def load_arena_data(args) -> Tuple[pd.DataFrame, Callable, str]:
    """Load and preprocess the standard arena dataset."""
    print("Loading arena dataset…")
    dataset = load_dataset("lmarena-ai/arena-human-preference-100k", split="train")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} battles from arena dataset")

    if getattr(args, "filter_english", False):
        df = df[df["language"] == "English"]
        print(f"After English filter: {len(df)} battles")

    models = [
        "claude-3-5-sonnet-20240620",
        "gpt-4o-2024-05-13",
        "gemini-1.5-pro-api-0514",
        "llama-3-70b-instruct",
        "gemini-1.5-pro-exp-0801",
        "claude-3-opus-20240229",
        "llama-3.1-405b-instruct",
        "chatgpt-4o-latest",
        "gpt-4-turbo-2024-04-09",
        "deepseek-v2-api-0628",
        "gpt-4o-2024-08-06",
    ]
    df = df[df["model_a"].isin(models) & df["model_b"].isin(models)]
    print(f"After model filter: {len(df)} battles")

    def parse_winner(row):
        if row["winner"] == "model_a":
            return row["model_a"]
        elif row["winner"] == "model_b":
            return row["model_b"]
        else:
            return row["winner"]
    
    df["winner"] = df.apply(parse_winner, axis=1)
    df["score"] = df.apply(lambda row: {"winner": row["winner"]}, axis=1)

    df = df.dropna(subset=["conversation_a", "conversation_b"])
    print(f"After removing missing conversations: {len(df)} battles")

    # ------------------------------------------------------------------
    # Extract user prompt and both model responses into explicit columns
    # ------------------------------------------------------------------
    def _extract_responses(row):
        user_prompt, model_a_resp = _extract_content_arena(row["conversation_a"])
        _, model_b_resp = _extract_content_arena(row["conversation_b"])
        return pd.Series({
            "prompt": user_prompt,
            "model_a_response": model_a_resp,
            "model_b_response": model_b_resp,
        })

    response_df = df.apply(_extract_responses, axis=1)
    df = pd.concat([df, response_df], axis=1)
   
    if "__index_level_0__" in df.columns:
        df = df.drop(columns="__index_level_0__")

    # Deduplicate now that prompt column definitely exists
    if "prompt" in df.columns:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["prompt"])
        print(f"After removing duplicates: {before_dedup - len(df)} rows dropped, {len(df)} remain")

    return df, _extract_content_arena, "one_sided_system_prompt_no_examples"

def load_arena_data_single(args) -> Tuple[pd.DataFrame, Callable, str]:
    """Load and preprocess the standard arena dataset."""
    df, _, _ = load_arena_data(args)

    df_a = df.copy()
    df_a["model"] = df['model_a']
    df_a["model_response"] = df['model_a_response']
    df_b = df.copy()
    df_b["model"] = df['model_b']
    df_b["model_response"] = df['model_b_response']
    df = pd.concat([df_a, df_b])
    df = df.drop(columns=["model_a", "model_a_response", "model_b", "model_b_response", "score"])
    df = df.dropna(subset=["model", "model_response"])
    print(f"After removing missing model and model response: {len(df)} battles")
    print(df.columns)
    
    return df, _extract_content_arena, "single_model_system_prompt"
    

# ---------------------------------------------------------------------------
# Web-dev loaders
# ---------------------------------------------------------------------------

def load_webdev_data(args):
    """Load and preprocess the web-dev arena dataset."""
    print("Loading webdev arena dataset…")
    dataset = load_dataset("lmarena-ai/webdev-arena-preference-10k", split="test")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} battles from webdev arena dataset")

    if getattr(args, "filter_english", False):
        df = df[df["language"] == "English"]
        print(f"After English filter: {len(df)} battles")

    if getattr(args, "exclude_ties", False):
        df = df[~df["winner"].str.contains("tie")]
        print(f"After excluding ties: {len(df)} battles")

    df = df.dropna(subset=["conversation_a", "conversation_b"])
    df["prompt"] = df.conversation_a.apply(lambda x: x[0]["content"][0]["text"])
    print(f"After extracting prompt: {len(df)} battles")

    def parse_winner(row):
        if row["winner"] == "model_a":
            return row["model_a"]
        elif row["winner"] == "model_b":
            return row["model_b"]
        else:
            return row["winner"]
    
    df["winner"] = df.apply(parse_winner, axis=1)
    df["score"] = df.apply(lambda row: {"winner": row["winner"]}, axis=1)

    # Extract user prompt and both model responses
    def _extract_responses_webdev(row):
        user_prompt, model_a_resp = _extract_content_webdev(row["conversation_a"])
        _, model_b_resp = _extract_content_webdev(row["conversation_b"])
        return pd.Series({
            "prompt": user_prompt,
            "model_a_response": model_a_resp,
            "model_b_response": model_b_resp,
        })

    response_df = df.apply(_extract_responses_webdev, axis=1)
    df = pd.concat([df, response_df], axis=1)

    # Deduplicate now that prompt column definitely exists
    if "prompt" in df.columns:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["prompt"])
        print(f"After removing duplicates: {before_dedup - len(df)} rows dropped, {len(df)} remain")

    return df, _extract_content_webdev, "webdev_system_prompt_no_examples"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_LOADERS: Dict[str, Callable] = {
    "arena": load_arena_data,
    "webdev": load_webdev_data,
    "arena_single": load_arena_data_single,
}


def load_data(dataset_name: str, args):
    """Dispatch helper selecting the right dataset loader."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_LOADERS)}"
        )
    return DATASET_LOADERS[dataset_name](args) 