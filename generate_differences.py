import argparse
import os
import json
import pandas as pd
import wandb
from transformers import AutoTokenizer
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np

from prompts import (
    one_sided_system_prompt,
    webdev_system_prompt,
    one_sided_system_prompt_no_examples,
    webdev_system_prompt_no_examples,
    coding_system_prompt,
    coding_system_prompt_no_examples,
    fictional_system_prompt,
)
from data_loader import load_data


def format_example(
    user_prompt, model_a_name, model_a_response, model_b_name, model_b_response
):
    """Format the arena data into the expected format for the LLM."""
    return f"Prompt: {user_prompt}\n{model_a_name} response: {model_a_response}\n\n{model_b_name} response: {model_b_response}"


def remove_thinking_from_output(output):
    """Remove thinking tags from model output."""
    if pd.isna(output) or not isinstance(output, str):
        return str(output) if not pd.isna(output) else ""

    if "</think>" in output:
        return output.split("</think>")[1]
    else:
        return output


def parse_json_response(response_text):
    """Parse JSON response from model, handling potential formatting issues."""
    try:
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_content = response_text[json_start:json_end].strip()
        else:
            json_content = response_text.strip()
        parsed_json = json.loads(json_content)
        return parsed_json, None
    except Exception as e:
        return None, str(e)


def call_openai_api(message, model_name, temperature, top_p, max_tokens, system_prompt):
    """Call OpenAI API using LiteLLM with error handling."""
    try:
        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            top_p=top_p,
            caching=True,
        )
        return response.choices[0].message.content
    except litellm.ContextWindowExceededError as e:
        print(f"Caught ContextWindowExceededError: {e}")
        return "ERROR: Context window exceeded. Input is too long."
    except Exception as e:
        if (
            "context_length_exceeded" in str(e).lower()
            or "input is too long" in str(e).lower()
        ):
            print(f"Caught context length error in generic exception: {e}")
            return "ERROR: Context window exceeded. Input is too long."
        print(f"Error calling OpenAI API: {str(e)}")
        return f"ERROR: {str(e)}"


def process_openai_batch(
    messages, model_name, temperature, top_p, max_tokens, system_prompt, max_workers=10
):
    """Process a batch of messages using OpenAI API with threading."""
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                call_openai_api,
                msg,
                model_name,
                temperature,
                top_p,
                max_tokens,
                system_prompt,
            ): i
            for i, msg in enumerate(messages)
        }
        results = [None] * len(messages)
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing message {index}: {str(e)}")
                results[index] = f"ERROR: {str(e)}"
        responses = results
    return responses


def save_intermediate_results(results_df, output_file, batch_num):
    """Save intermediate results to the main output file and log to wandb."""
    results_df.to_csv(output_file, index=False)
    print(f"Updated results saved to {output_file} (after batch {batch_num})")
    
    batch_size = 10
    start_idx = max(0, (batch_num - 1) * batch_size)
    end_idx = min(len(results_df), batch_num * batch_size)
    
    current_batch_results = results_df.iloc[start_idx:end_idx]
    
    wandb.log({
        "batch_completed": batch_num,
        "total_processed": len(results_df),
        f"batch_{batch_num}_results": wandb.Table(dataframe=current_batch_results.astype(str)),
        "all_results_so_far": wandb.Table(dataframe=results_df.astype(str))
    })


def save_batch_requests(requests, metadata_list, output_file, metadata_file):
    """Save batch requests to JSONL file and metadata to separate JSON file."""
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(metadata_file) if os.path.dirname(metadata_file) else '.', exist_ok=True)

    with open(output_file, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    def _sanitize(obj):
        """Recursively convert objects to JSON-serializable formats."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_sanitize(v) for v in obj]
        # numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        # pandas timestamp
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # fallback: convert to string
        return str(obj)

    serializable_metadata = [_sanitize(m) for m in metadata_list]

    with open(metadata_file, "w") as f:
        json.dump(serializable_metadata, f, indent=2)

    print(f"Saved {len(requests)} batch requests to {output_file}")
    print(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model responses from a dataset and identify differences."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["arena", "webdev", "coding", "fictional"],
        help="Name of the dataset to process.",
    )
    parser.add_argument("--num_samples", type=int, help="Number of battles to process.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model name for analysis.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file. Defaults to differences/[dataset_name]_differences.csv",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size."
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p.")
    parser.add_argument("--max_tokens", type=int, default=16000, help="Max tokens.")
    parser.add_argument(
        "--max_model_len", type=int, default=32000, help="Max model length."
    )
    parser.add_argument(
        "--filter_english",
        action="store_true",
        help="Filter to only English conversations.",
    )
    parser.add_argument(
        "--exclude_ties", action="store_true", help="Exclude tied battles."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Max threads for OpenAI API calls.",
    )
    parser.add_argument(
        "--batches",
        action="store_true",
        help="If set, skip generation/parsing and instead create OpenAI Batch API request (.jsonl) and metadata (.json) files.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Metadata file path when using --batches flag. Defaults to <output_file> with _metadata.json suffix.",
    )
    args = parser.parse_args()

    # Determine default output locations depending on mode
    if args.batches:
        if args.output_file is None:
            args.output_file = f"batches/{args.dataset}_batch_requests.jsonl"
        if args.metadata_file is None:
            # Save alongside the requests file
            base, _ = os.path.splitext(args.output_file)
            args.metadata_file = base + "_metadata.json"
    else:
        if args.output_file is None:
            args.output_file = f"differences/{args.dataset}_differences.csv"

    SYSTEM_PROMPTS = {
        "one_sided_system_prompt": one_sided_system_prompt,
        "one_sided_system_prompt_no_examples": one_sided_system_prompt_no_examples,
        "webdev_system_prompt": webdev_system_prompt,
        "webdev_system_prompt_no_examples": webdev_system_prompt_no_examples,
        "coding_system_prompt": coding_system_prompt,
        "coding_system_prompt_no_examples": coding_system_prompt_no_examples,
        "fictional_system_prompt": fictional_system_prompt,
    }

    df, extract_content_from_conversation, system_prompt_name = load_data(
        args.dataset, args
    )
    system_prompt = SYSTEM_PROMPTS[system_prompt_name]

    run = wandb.init(
        project="arena-difference-training", name=f"{args.dataset}_data_gen"
    )
    run.summary["system_prompt"] = system_prompt
    run.config.update(vars(args))
    run.config["system_prompt_type"] = system_prompt_name
    run.config["dataset"] = args.dataset

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if args.num_samples and args.num_samples < len(df):
        df = df.head(args.num_samples)
        print(f"Limited to {args.num_samples} battles")

    # ------------------------------------------------------------------
    # Batch mode: generate OpenAI Batch API request & metadata files only
    # ------------------------------------------------------------------
    if args.batches:
        print(f"Batch mode enabled: generating request file {args.output_file} and metadata file {args.metadata_file} …")

        batch_requests = []
        batch_metadata = []
        processed_count = 0
        skipped_count = 0

        for _, row in df.iterrows():
            try:
                user_prompt_a, model_a_response = extract_content_from_conversation(row['conversation_a'])
                _, model_b_response = extract_content_from_conversation(row['conversation_b'])

                # If any essential component is missing, skip
                if not user_prompt_a or not model_a_response or not model_b_response:
                    skipped_count += 1
                    continue

                user_prompt = user_prompt_a
                row_qid = row['question_id'] if 'question_id' in row else row.name
                custom_id = f"{args.dataset}-comparison-{row_qid}"

                formatted_input = format_example(
                    user_prompt,
                    row['model_a'],
                    model_a_response,
                    row['model_b'],
                    model_b_response,
                )

                # Compose batch request entry
                batch_request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": formatted_input},
                        ],
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                    },
                }

                batch_requests.append(batch_request)

                # Convert pandas row to dict and ensure all values are JSON serializable
                meta_row = {}
                for key, value in row.items():
                    if isinstance(value, (np.ndarray, np.generic)):
                        if isinstance(value, np.ndarray):
                            meta_row[key] = value.tolist()
                        else:
                            meta_row[key] = value.item()
                    elif pd.isna(value):
                        meta_row[key] = None
                    else:
                        meta_row[key] = value
                
                meta_row.update(
                    {
                        "custom_id": custom_id,
                        "user_prompt": user_prompt,
                        "model_a_response": model_a_response,
                        "model_b_response": model_b_response,
                    }
                )
                batch_metadata.append(meta_row)
                processed_count += 1
            except Exception as e:
                print(
                    f"Error processing row with question_id {row.get('question_id', 'unknown')}: {e}"
                )
                skipped_count += 1

        # Save to disk
        save_batch_requests(batch_requests, batch_metadata, args.output_file, args.metadata_file)

        print(
            f"Batch generation complete. Processed {processed_count} rows, skipped {skipped_count}. "
            f"Total requests written: {len(batch_requests)}"
        )

        wandb.log(
            {
                "total_battles_processed": processed_count,
                "battles_skipped": skipped_count,
                "batch_requests_generated": len(batch_requests),
            }
        )
        wandb.finish()
        return  # Exit early – no model inference or parsing

    use_openai = args.model_name.lower().startswith("gpt")
    
    if use_openai:
        print(f"Using OpenAI API for model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        llm = None
        sampling_params = None
    else:
        print(f"Initializing vLLM: {args.model_name}")
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is required for non-OpenAI models. Install with: pip install vllm")
        
        llm = LLM(
            model=args.model_name,
            tokenizer=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    
    # new_data = {
    #     "question_id": [], "user_prompt": [], "model_a_name": [], "model_b_name": [],
    #     "model_a_response": [], "model_b_response": [], "winner": [], "differences": [],
    #     "parsed_differences": [], "parse_error": []
    # }
    keys = df.columns.tolist()
    new_data = {
        key: [] for key in keys + ["differences", "parsed_differences", "parse_error"]
    }
    new_data["differences"] = []
    new_data["parsed_differences"] = []
    new_data["parse_error"] = []

    batch_size = 100
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        # batch_messages, batch_data = [], {
        #     "question_id": [], "user_prompt": [], "model_a_name": [], "model_b_name": [],
        #     "model_a_response": [], "model_b_response": [], "winner": []
        # }
        batch_messages, batch_data = [], {key: [] for key in keys}
        # add in "user_prompt", "model_a_response", "model_b_response"
        batch_data["user_prompt"] = []
        batch_data["model_a_response"] = []
        batch_data["model_b_response"] = []
        
        print(f"Processing batch {batch_start//batch_size + 1}: battles {batch_start+1}-{batch_end}")
        
        for _, row in batch_df.iterrows():
            try:
                user_prompt_a, model_a_response = extract_content_from_conversation(row['conversation_a'])
                _, model_b_response = extract_content_from_conversation(row['conversation_b'])
                
                formatted_input = format_example(
                    user_prompt_a, row['model_a'], model_a_response, row['model_b'], model_b_response
                )
                
                if use_openai:
                    batch_messages.append(formatted_input)
                else:
                    message = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": formatted_input}
                        ],
                        tokenize=False, add_generation_prompt=True,
                    )
                    batch_messages.append(message)
                
                # Store the data for this battle
                # add in all the rows from the dataframe
                for key in row.index:
                    if key in batch_data:
                        batch_data[key].append(row[key])
                    else:
                        batch_data[key] = [row[key]]

                batch_data["user_prompt"].append(user_prompt_a)
                batch_data["model_a_response"].append(model_a_response)
                batch_data["model_b_response"].append(model_b_response)

            except Exception as e:
                print(f"Error processing row with question_id {row.get('question_id', 'unknown')}: {e}")
                continue
        
        if not batch_messages:
            continue
        
        if use_openai:
            batch_responses = process_openai_batch(
                batch_messages, args.model_name, args.temperature, args.top_p,
                args.max_tokens, system_prompt, args.max_workers
            )
        else:
            vllm_responses = llm.generate(batch_messages, sampling_params)
            batch_responses = [resp.outputs[0].text for resp in vllm_responses]
        
        for i, response in enumerate(batch_responses):
            cleaned_response = remove_thinking_from_output(response) if not use_openai else response
            parsed_json, parse_error = parse_json_response(cleaned_response)
            
            # We need to get the correct item from batch_data
            original_data_row = batch_data
            
            for key in batch_data:
                if key in new_data:
                    new_data[key].append(batch_data[key][i])
                else:
                    new_data[key] = [batch_data[key][i]]
            new_data["differences"].append(cleaned_response)
            new_data["parsed_differences"].append(parsed_json)
            new_data["parse_error"].append(parse_error)
            
        current_results_df = pd.DataFrame(new_data)
        save_intermediate_results(current_results_df, args.output_file, batch_start//batch_size + 1)
        
        successful_parses = sum(1 for pe in new_data["parse_error"] if pe is None)
        total_processed = len(new_data["parse_error"])
        parsing_rate = successful_parses/total_processed*100 if total_processed > 0 else 0
        wandb.log({"parsing_success_rate": parsing_rate})

    final_df = pd.DataFrame(new_data)
    if not final_df.empty:
        final_df.to_json(args.output_file.replace(".csv", ".jsonl"), orient='records', lines=True)
        run.log({"final_results": wandb.Table(dataframe=final_df.astype(str))})

    wandb.finish()

if __name__ == "__main__":
    main() 