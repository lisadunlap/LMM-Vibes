import pandas as pd
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process and analyze model comparison data')
    parser.add_argument('--input_file', type=str, default="differences/test.jsonl",
                       help='Input JSONL file to process')
    return parser.parse_args()

def remove_things(x):
    x = x[x.find('_')+1:]
    x = x.replace("-Instruct", "")
    return x.lower()

def model_name_pass(row):
    model_a = row["model_1_name"]
    model_b = row["model_2_name"]
    model_a_modified_name = remove_things(model_a)
    model_b_modified_name = remove_things(model_b)
    model = row["model"]
    if model == model_a or model.lower() == "model a" or remove_things(model) == model_a_modified_name:
        return model_a
    if model == model_b or model.lower() == "model b" or remove_things(model) == model_b_modified_name:
        return model_b
    return None

def main():
    args = parse_args()
    
    # Initialize wandb
    run = wandb.init(project="arena-difference-training", name="post_processing")

    # Load and process data
    df = pd.read_json(args.input_file, lines=True)
    new_df = df.explode("parsed_differences")
    new_df.rename(columns={
        "model_a_name": "model_1_name",
        "model_b_name": "model_2_name",
        "model_a_response": "model_1_response",
        "model_b_response": "model_2_response"
    }, inplace=True)
    new_df = new_df[new_df["parsed_differences"].apply(lambda x: isinstance(x, dict))]
    behaviors = pd.DataFrame(new_df["parsed_differences"].tolist())

    # Create final dataframe
    total_df = new_df.reset_index(drop=True)
    behaviors = behaviors.reset_index(drop=True)
    total_df = pd.concat([total_df, behaviors], axis=1)
    total_df["model"] = total_df.apply(model_name_pass, axis=1)

    # Log to wandb and save
    wandb.log({"processed_data": wandb.Table(dataframe=total_df.astype(str))})
    total_df.to_json(args.input_file.replace('.jsonl', '_processed.jsonl'), orient="records", lines=True)
    
    wandb.finish()

if __name__ == "__main__":
    main()