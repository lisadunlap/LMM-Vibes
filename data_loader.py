import pandas as pd
from datasets import load_dataset
import ast
import re


def _extract_content_arena(conversation):
    """Extract content for standard arena data."""
    # If conversation is a string, parse it as Python literal first
    if isinstance(conversation, str):
        # Fix malformed syntax: replace newline+space between dict entries with comma
        conversation = re.sub(r'}\s*\n\s*{', '}, {', conversation)
        conversation = ast.literal_eval(conversation)
    
    return conversation[0]["content"], conversation[1]["content"]


selected_keys = ['code', 'commentary', 'description', 'file_path', 'has_additional_dependencies', 'additional_dependencies', 'install_dependencies_command', 'port', 'template', 'title']
def _extract_content_webdev(conversation):
    """Extract content for webdev arena data."""
    formatted_object = ""
    for key in selected_keys:
        formatted_object += f"## {key}\n{conversation[1]['object'][key]}\n\n"
    formatted_response = f"## Text response\n{conversation[1]['content'][0]['text']}\n\n{formatted_object}## Logs\n{conversation[1]['result']}"
    return conversation[0]["content"], formatted_response


def load_arena_data(args):
    """Loads and preprocesses the standard arena dataset."""
    print("Loading arena dataset...")
    dataset = load_dataset("lmarena-ai/arena-human-preference-100k", split="train")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} battles from arena dataset")

    if args.filter_english:
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

    df = df.dropna(subset=["conversation_a", "conversation_b"])
    print(f"After removing missing conversations: {len(df)} battles")

    return df, _extract_content_arena, "one_sided_system_prompt"


def load_webdev_data(args):
    """Loads and preprocesses the webdev arena dataset."""
    print("Loading webdev arena dataset...")
    dataset = load_dataset("lmarena-ai/webdev-arena-preference-10k", split="test")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} battles from webdev arena dataset")

    if args.filter_english:
        df = df[df["language"] == "English"]
        print(f"After English filter: {len(df)} battles")

    if args.exclude_ties:
        df = df[~df["winner"].str.contains("tie")]
        print(f"After excluding ties: {len(df)} battles")

    df = df.dropna(subset=["conversation_a", "conversation_b"])
    print(f"After removing missing conversations: {len(df)} battles")

    return df, _extract_content_webdev, "webdev_system_prompt_no_examples"

def load_arena_coding_data(args):
    """Loads and preprocesses the coding arena dataset."""
    print("Loading coding arena dataset...")
    df = pd.read_csv("out/filtered_conversations_and_category.csv")
    print(f"Loaded {len(df)} battles from coding arena dataset")
    df = df[df['narrower_category_id'] == 0] 
    return df, _extract_content_arena, "coding_system_prompt"


def load_arena_fictional_data(args):
    """Loads and preprocesses the fictional arena dataset."""
    print("Loading fictional arena dataset...")
    df = pd.read_csv("out/filtered_conversations_and_category.csv")
    print(f"Loaded {len(df)} battles from fictional arena dataset")
    df = df[df['narrower_category_id'] == 1] 
    return df, _extract_content_arena, "fictional_system_prompt"

DATASET_LOADERS = {"arena": load_arena_data, "webdev": load_webdev_data, "coding": load_arena_coding_data, "fictional": load_arena_fictional_data}


def load_data(dataset_name, args):
    """Load data based on dataset name."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available datasets are: {list(DATASET_LOADERS.keys())}"
        )
    return DATASET_LOADERS[dataset_name](args)
