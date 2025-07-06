"""Load the first 10 rows of the arena dataset and make sure the DataFrame
has the expected columns.  This uses the helper in data_loader.py.

Note: The Hugging Face dataset download can be slow the first time.  For a
quick unit-test environment you might want to mock `datasets.load_dataset`.
"""

from types import SimpleNamespace

import pandas as pd
from lmmvibes.extractors.openai import OpenAIExtractor
from lmmvibes.postprocess.parser import LLMJsonParser
from lmmvibes.datasets import load_data
from lmmvibes.core.data_objects import PropertyDataset


def test_first_10_arena_rows():
    # Build a minimal args namespace expected by dataset loaders
    args = SimpleNamespace(filter_english=False)

    df, extract_content_fn, _ = load_data("arena", args)
    first10 = df.head(50)

    # Basic sanity checks
    assert len(first10) == 50
    required_cols = {"prompt", "conversation_a", "conversation_b", "model_a", "model_b", "model_a_response", "model_b_response"}
    assert required_cols.issubset(first10.columns)

    print("Loaded first 50 arena rows with columns:", list(first10.columns))

    # get properties
    dataset = PropertyDataset.from_dataframe(first10, method="side_by_side")
    print("..done loading properties")

    # ------------------------------------------------------------------
    # Extract, parse and save results WITH wandb (preferred path)
    # ------------------------------------------------------------------
    import wandb
    wandb.init(project="lmm-vibes-test", name="test_run")

    extractor = OpenAIExtractor(verbose=False, use_wandb=True)
    parser = LLMJsonParser(verbose=False, use_wandb=True)

    dataset_after_extract = extractor(dataset)
    dataset_after_parse = parser(dataset_after_extract)
    # print(f"Dataset after parse: {dataset_after_parse}")

    # ------------------------------------------------------------
    # Save results for downstream clustering / analysis
    # ------------------------------------------------------------
    import pathlib
    output_dir = pathlib.Path("tests/outputs")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "arena_first50_dataset.json"
    parquet_path = output_dir / "arena_first50_properties.parquet"

    dataset_after_parse.save(str(json_path), format="json")
    dataset_after_parse.to_dataframe().to_parquet(parquet_path, index=False)

    print(f"Saved parsed dataset to {json_path} and {parquet_path}")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    test_first_10_arena_rows()
    print("✅ Arena subset test passed!") 