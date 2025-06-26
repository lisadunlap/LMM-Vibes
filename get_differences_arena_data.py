import re
import argparse
import os
import json
from vllm import LLM, SamplingParams
import pandas as pd
import wandb
from transformers import AutoTokenizer
from datasets import load_dataset
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# Use the same system prompt as the original script
one_sided_system_prompt = """You are an expert model behavior analyst. Your task is to meticulously compare two model responses to a given user prompt and identify unique qualitative properties belonging to one model but not the other. For each significant property, you must determine if it's more likely a **general trait** of the model or a **context-specific** behavior triggered by the current prompt.

**Prioritize conciseness and clarity in all your descriptions and explanations.** Aim for the most impactful information in the fewest words.

You will be provided with:
1.  **User Prompt:** The original prompt given to both models.
2.  **Model A Name:** The identifier for Model A.
3.  **Model A Response:** The response from Model A.
4.  **Model B Name:** The identifier for Model B.
5.  **Model B Response:** The response from Model B.

**Your Goal:**
Produce a JSON list of objects. Each object will represent a single distinct property observed in one model's response that is notably absent or different in the other's. Focus on identifying key areas of distinction, and the individual property observations in the output list (e.g., Model A's formal tone would be one entry, Model B's casual tone would be another related entry). As these are very common and easy to measure with heuristics, please do not include properties like "Model A is more concise than Model B". If applicatble, make sure to also include properties revolving around the models reasoning, interpretation of the prompt/intent, and potential reason for errors if they exist. 

**Definitions:**
*   **General Trait:** Reflects a model's typical behavior across diverse prompts.
    *   *Think:* Is this how this Model *usually* is compared to the other?
*   **Context-Specific Difference:** Arises mainly due to *this specific* user prompt.
    *   *Think:* Is this property a direct reaction to *this current prompt*?
*   **Impact:** How much does this property impact the user's experience?
    *   *Think:* Is this property a major factor in the user's experience? Note that this could depend on the user's intent and the context of the prompt.
*   **Contains Errors:** Does either model response contain errors?
    *   *Think:* Are there factual errors, hallucinations, or other strange or unwanted behavior?
*   **Unexpected Behavior:** Does the model's response contain highly unusual or concerning behavior? If true then a developer will analyze these responses manually.
    *   *Think:* Does this involve offensive language, gibberish, bias, factual hallucinations, or other strange or unwanted behavior?

**JSON Output Structure for each property (BE BRIEF, if no notable properties exist, return empty list. Please use the names of the models in the output rather than "Model A"/"Model B"):**
```json
[
  {
    "model": "Model A|Model B",
    "property_description": "Brief description of the unique property observed in this model (max 2 sentences)",
    "category": "1-4 word category",
    "evidence": "Direct quote or evidence from the specified model",
    "type": "General|Context-Specific",
    "reason": "Brief justification for this property, noting its absence/difference in the other model (max 2 sentences)",
    "impact": "Low|Medium|High",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False"
  }
]
```

**Example JSON Output (Note: This is a simplified example and does not include all possible properties):**
```json
[
  {
    "model": "{{Model A Name}}",
    "property_description": "formal and professional tone.",
    "category": "Tone",
    "evidence": "Quote: 'It is imperative to consider the implications...'",
    "type": "General",
    "reason": "{{Model A Name}}'s response is in a formal register, which is a notable contrast to {{Model B Name}}'s more casual style.",
    "impact": "Low",
    "contains_errors": "False",
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model B Name}}",
    "property_description": "casual and conversational tone.",
    "category": "Tone",
    "evidence": "Quote: 'Hey there! So, basically, what you gotta think about is...'",
    "type": "General",
    "reason": "{{Model B Name}}'s response is in an informal, friendly style, which stands out compared to {{Model A Name}}'s formality.",
    "impact": "Low",
    "contains_errors": "False",
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model A Name}}",
    "property_description": "functional programming approach.",
    "category": "Coding Style",
    "evidence": "Uses `map()` and `filter()` functions extensively for data transformation.",
    "type": "Context-Specific",
    "reason": "For this data processing task, {{Model A Name}} opted for a functional approach, which was not seen in {{Model B Name}}'s object-oriented solution.",
    "impact": "High",
    "contains_errors": "False",
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model B Name}}",
    "property_description": "object-oriented programming approach.",
    "category": "Coding Style",
    "evidence": "Defines a `DataProcessor` class with methods like `process()` and `validate()`.",
    "type": "Context-Specific",
    "reason": "In response to the coding prompt, {{Model B Name}} chose an object-oriented design, contrasting with {{Model A Name}}'s functional implementation.",
    "impact": "High",
    "contains_errors": "False",
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model A Name}}",
    "property_description": "cautious approach to factual claims.",
    "category": "Fact Verification",
    "evidence": "Quote: 'According to the 2023 WHO report... However, this data may vary by region and should be cross-referenced.'",
    "type": "General",
    "reason": "{{Model A Name}} prioritizes accuracy and uncertainty, providing source attribution and disclaimers, unlike {{Model B Name}}'s direct factual statements.",
    "impact": "Medium",
    "contains_errors": "False",
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model B Name}}",
    "property_description": "factual information with high confidence and without explicit verification or caveats.",
    "category": "Fact Verification",
    "evidence": "Quote: 'The global vaccination rate is 78% and continues to increase rapidly worldwide.'",
    "type": "General",
    "reason": "{{Model B Name}} states flase facts without providing sources or acknowledging potential variability, contrasting with {{Model A Name}}'s cautious approach.",
    "impact": "High",
    "contains_errors": "True",
    "unexpected_behavior": "False"
  }
]
```"""

def extract_content_from_conversation(conversation):
    """Extract the user prompt and assistant response from conversation format."""
    
    return conversation[0]['content'], conversation[1]['content']

def format_arena_example(user_prompt, model_a_name, model_a_response, model_b_name, model_b_response):
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
        # Try to find JSON content between ```json and ``` or just parse directly
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_content = response_text[json_start:json_end].strip()
        else:
            # Try to find JSON array/object in the response
            json_content = response_text.strip()
        
        # Parse the JSON
        parsed_json = json.loads(json_content)
        return parsed_json, None
    except Exception as e:
        return None, str(e)

def call_openai_api(message, model_name, temperature, top_p, max_tokens):
    """Call OpenAI API using LiteLLM with error handling."""
    try:
        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": one_sided_system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            caching=True
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return f"ERROR: {str(e)}"

def process_openai_batch(messages, model_name, temperature, top_p, max_tokens, max_workers=10):
    """Process a batch of messages using OpenAI API with threading."""
    responses = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(call_openai_api, msg, model_name, temperature, top_p, max_tokens): i 
            for i, msg in enumerate(messages)
        }
        
        # Collect results in order
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
    # Save to the main output file (overwrite with updated results)
    results_df.to_csv(output_file, index=False)
    print(f"Updated results saved to {output_file} (after batch {batch_num})")
    
    # Log to wandb - both incremental and cumulative views
    batch_size = 10  # Assuming 10 per batch
    start_idx = max(0, (batch_num - 1) * batch_size)
    end_idx = min(len(results_df), batch_num * batch_size)
    
    # Log just this batch's results
    current_batch_results = results_df.iloc[start_idx:end_idx]
    
    # Log both the current batch and cumulative results
    wandb.log({
        "batch_completed": batch_num,
        "total_processed": len(results_df),
        f"batch_{batch_num}_results": wandb.Table(dataframe=current_batch_results.astype(str)),  # This batch only
        "all_results_so_far": wandb.Table(dataframe=results_df.astype(str))  # All results accumulated
    })

def truncate_long_responses(user_prompt, model_a_response, model_b_response, tokenizer, max_model_len, system_prompt, model_a_name="ModelA", model_b_name="ModelB", safety_margin=500, use_openai=False):
    """
    Truncate model responses if the total prompt would exceed max_model_len.
    Returns tuple of (truncated_a_response, truncated_b_response, was_truncated)
    """
    def estimate_token_count(text, tokenizer, use_openai):
        """Estimate token count for the given text."""
        if use_openai:
            return len(text) // 4  # Rough estimate: 4 chars per token
        return len(tokenizer.encode(text))
    
    def create_full_message(user_prompt, model_a_response, model_b_response, system_prompt, use_openai):
        """Create the full message as it would be sent to the model."""
        formatted_input = format_arena_example(
            user_prompt, model_a_name, model_a_response, model_b_name, model_b_response
        )
        
        if use_openai:
            return system_prompt + "\n\n" + formatted_input
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": formatted_input}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # Check if truncation is needed
    test_message = create_full_message(
        user_prompt, model_a_response, model_b_response, system_prompt, use_openai
    )
    if estimate_token_count(test_message, tokenizer, use_openai) <= max_model_len:
        return model_a_response, model_b_response, False

    # Calculate available space for responses
    base_message = create_full_message(user_prompt, "", "", system_prompt, use_openai)
    base_tokens = estimate_token_count(base_message, tokenizer, use_openai)
    available_tokens = max(100, max_model_len - base_tokens - safety_margin)
    tokens_per_response = available_tokens // 2

    # Convert tokens to characters (rough estimate)
    chars_per_token = 4 if use_openai else 3
    max_chars = tokens_per_response * chars_per_token

    # Truncate responses
    truncated_a = model_a_response[:max_chars] + " [truncated]" if len(model_a_response) > max_chars else model_a_response
    truncated_b = model_b_response[:max_chars] + " [truncated]" if len(model_b_response) > max_chars else model_b_response

    return truncated_a, truncated_b, True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare model responses from arena dataset and identify differences')
    parser.add_argument('--num_samples', type=int,
                       help='Number of battles to process')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini",
                       help='Model name to use for the LLM analysis')
    parser.add_argument('--output_file', type=str, default="differences/arena_differences.csv",
                       help='Output file to save the results')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top p')
    parser.add_argument('--max_tokens', type=int, default=2048,
                       help='Max tokens')
    parser.add_argument('--max_model_len', type=int, default=16384,
                       help='Max model length')
    parser.add_argument('--filter_english', action='store_true', default=True,
                       help='Filter to only English conversations')
    parser.add_argument('--exclude_ties', action='store_true', default=True,
                       help='Exclude tied battles')
    parser.add_argument('--ties_only', action='store_true', default=False,
                       help='Only include tied battles')
    parser.add_argument('--auto_truncate', action='store_true', default=True,
                       help='Automatically truncate long prompts to fit within max_model_len')
    parser.add_argument('--truncation_safety_margin', type=int, default=1000,
                       help='Reserve this many tokens as safety margin when truncating (default: 1000)')
    parser.add_argument('--max_workers', type=int, default=16,
                       help='Maximum number of threads for OpenAI API calls (default: 10)')
    args = parser.parse_args()

    # Initialize wandb
    run = wandb.init(project="arena-difference-training", name="arena_data_gen")
    run.summary["system_prompt"] = one_sided_system_prompt
    run.config.update(vars(args))
    run.config["system_prompt_type"] = "one_sided_json_arena"

    # Make output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load the arena dataset
    print("Loading arena dataset...")
    dataset = load_dataset("lmarena-ai/arena-human-preference-100k", split="train")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} battles from arena dataset")

    # Apply filters
    if args.filter_english:
        df = df[df['language'] == 'English']
        print(f"After English filter: {len(df)} battles")

    # if args.ties_only:
    #     df = df[df['winner'].str.contains('tie', na=False)]
    #     print(f"After ties only filter: {len(df)} battles")
    # elif args.exclude_ties:
    #     df = df[~df['winner'].str.contains('tie', na=False)]
    #     print(f"After excluding ties: {len(df)} battles")
    
    models  = [
        'claude-3-5-sonnet-20240620',
        'gpt-4o-2024-05-13',
        'gemini-1.5-pro-api-0514',
        'llama-3-70b-instruct',
        'gemini-1.5-pro-exp-0801',
        'claude-3-opus-20240229',
        'llama-3.1-405b-instruct',
        'chatgpt-4o-latest',
        'gpt-4-turbo-2024-04-09',
        'deepseek-v2-api-0628',
        'gpt-4o-2024-08-06',
        ]
    
    df = df[df['model_a'].isin(models) & df['model_b'].isin(models)]
    print(f"After model filter: {len(df)} battles")

    # Remove rows with missing conversation data
    df = df.dropna(subset=['conversation_a', 'conversation_b'])
    print(f"After removing missing conversations: {len(df)} battles")

    # Limit to specified number of samples
    if args.num_samples and args.num_samples < len(df):
        df = df.head(args.num_samples)
        print(f"Limited to {args.num_samples} battles")

    # Check if using OpenAI model
    use_openai = args.model_name.lower().startswith('gpt')
    
    if use_openai:
        print(f"Using OpenAI API for model: {args.model_name}")
        print(f"Threading enabled with max_workers: {args.max_workers}")
        # For OpenAI, we'll use a basic tokenizer for truncation estimation
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer as approximation
        llm = None  # No vLLM instance needed
        sampling_params = None
    else:
        # Initialize LLM for non-OpenAI models
        print(f"Initializing vLLM: {args.model_name}")
        llm = LLM(
            model=args.model_name,
            tokenizer=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
        )

        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Process battles in batches
    new_data = {
        "question_id": [], 
        "user_prompt": [], 
        "model_a_name": [], 
        "model_b_name": [], 
        "model_a_response": [], 
        "model_b_response": [], 
        "winner": [],
        "differences": [], 
        "parsed_differences": [], 
        "parse_error": []
    }

    batch_size = 100
    processed_count = 0
    truncated_count = 0
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        batch_messages = []
        batch_data = {
            "question_id": [],
            "user_prompt": [], 
            "model_a_name": [], 
            "model_b_name": [], 
            "model_a_response": [], 
            "model_b_response": [], 
            "winner": []
        }
        
        print(f"Processing batch {batch_start//batch_size + 1}: battles {batch_start+1}-{batch_end}")
        
        # Prepare batch data
        for _, row in batch_df.iterrows():
            try:
                # Extract conversation data
                user_prompt_a, model_a_response = extract_content_from_conversation(row['conversation_a'])
                user_prompt_b, model_b_response = extract_content_from_conversation(row['conversation_b'])
            
                # The user prompts should be the same, use the first one
                user_prompt = user_prompt_a
                model_a_name = row['model_a']
                model_b_name = row['model_b']
                
                # Truncate responses if necessary
                if args.auto_truncate:
                    truncated_a_response, truncated_b_response, was_truncated = truncate_long_responses(
                        user_prompt, model_a_response, model_b_response, tokenizer, args.max_model_len, one_sided_system_prompt, model_a_name, model_b_name, args.truncation_safety_margin, use_openai
                    )
                    if was_truncated:
                        truncated_count += 1
                else:
                    truncated_a_response, truncated_b_response = model_a_response, model_b_response
                
                # Create the formatted input for the LLM
                formatted_input = format_arena_example(
                    user_prompt, model_a_name, truncated_a_response, model_b_name, truncated_b_response
                )
                # print(formatted_input)
                
                if use_openai:
                    # For OpenAI, we just need the formatted input, not the full chat template
                    batch_messages.append(formatted_input)
                else:
                    # For vLLM, apply the chat template
                    message = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": one_sided_system_prompt}, 
                            {"role": "user", "content": formatted_input}
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    batch_messages.append(message)
                
                # Store the data for this battle
                batch_data["question_id"].append(row['question_id'])
                batch_data["user_prompt"].append(user_prompt)
                batch_data["model_a_name"].append(model_a_name)
                batch_data["model_b_name"].append(model_b_name)
                batch_data["model_a_response"].append(truncated_a_response)
                batch_data["model_b_response"].append(truncated_b_response)
                batch_data["winner"].append(row['winner'])
                
            except Exception as e:
                print(f"Error processing row with question_id {row.get('question_id', 'unknown')}: {str(e)}")
                print(f"Skipping this row and continuing...")
                continue
        
        if not batch_messages:
            print(f"No valid battles in batch {batch_start//batch_size + 1}, skipping...")
            continue
        
        # Safety check: validate all messages are within token limit before generation
        valid_messages = []
        valid_indices = []
        for i, message in enumerate(batch_messages):
            if use_openai:
                # For OpenAI, estimate token count more simply (4 chars per token)
                full_message = one_sided_system_prompt + "\n\n" + message
                token_count = len(full_message) // 4
            else:
                token_count = len(tokenizer.encode(message))
            
            if token_count <= args.max_model_len:
                valid_messages.append(message)
                valid_indices.append(i)
            else:
                print(f"Warning: Skipping message {i} with ~{token_count} tokens (exceeds {args.max_model_len})")
        
        if not valid_messages:
            print(f"No valid messages in batch {batch_start//batch_size + 1} after token validation, skipping...")
            continue
        
        print(f"Processing {len(valid_messages)} out of {len(batch_messages)} messages in batch")
        
        # Get LLM responses for the batch
        print(f"Getting LLM responses for {len(valid_messages)} battles...")
        
        if use_openai:
            # Use OpenAI API with threading
            batch_responses = process_openai_batch(
                valid_messages, 
                args.model_name, 
                args.temperature, 
                args.top_p, 
                args.max_tokens,
                args.max_workers
            )
        else:
            # Use vLLM
            vllm_responses = llm.generate(valid_messages, sampling_params=sampling_params)
            batch_responses = [resp.outputs[0].text for resp in vllm_responses]
        
        # Process the responses (only for valid indices)
        for response_idx, response in enumerate(batch_responses):
            original_idx = valid_indices[response_idx]  # Map back to original index
            if original_idx >= len(batch_data["question_id"]):  # Safety check
                break
                
            cleaned_response = remove_thinking_from_output(response) if not use_openai else response
            
            parsed_json, parse_error = parse_json_response(cleaned_response)
            
            # Add to main data structure
            new_data["question_id"].append(batch_data["question_id"][original_idx])
            new_data["user_prompt"].append(batch_data["user_prompt"][original_idx])
            new_data["model_a_name"].append(batch_data["model_a_name"][original_idx])
            new_data["model_b_name"].append(batch_data["model_b_name"][original_idx])
            new_data["model_a_response"].append(batch_data["model_a_response"][original_idx])
            new_data["model_b_response"].append(batch_data["model_b_response"][original_idx])
            new_data["winner"].append(batch_data["winner"][original_idx])
            new_data["differences"].append(cleaned_response)
            new_data["parsed_differences"].append(parsed_json)
            new_data["parse_error"].append(parse_error)
            
            processed_count += 1
        
        # Create current results dataframe and save after each batch
        current_results_df = pd.DataFrame(new_data)
        save_intermediate_results(current_results_df, args.output_file, batch_start//batch_size + 1)
        
        # Log successful parsing rate
        successful_parses = sum(1 for pe in new_data["parse_error"] if pe is None)
        total_processed = len(new_data["parse_error"])
        parsing_rate = successful_parses/total_processed*100 if total_processed > 0 else 0
        truncation_rate = truncated_count/total_processed*100 if total_processed > 0 else 0
        print(f"JSON parsing success rate so far: {successful_parses}/{total_processed} ({parsing_rate:.1f}%)")
        print(f"Truncation rate so far: {truncated_count}/{total_processed} ({truncation_rate:.1f}%)")
        print(f"Total battles processed: {processed_count}")
        
        # Log parsing stats to wandb
        wandb.log({
            "parsing_success_rate": parsing_rate,
            "successful_parses": successful_parses,
            "total_processed": total_processed,
            "battles_processed": processed_count,
            "truncation_rate": truncation_rate,
            "truncated_battles": truncated_count
        })

    # Create the final dataframe
    results_df = pd.DataFrame(new_data).dropna(subset=['question_id'])

    # Print sample results
    if len(results_df) > 0:
        print("\nSample raw response:")
        print(results_df["differences"].iloc[0])
        print("\nSample parsed JSON:")
        print(results_df["parsed_differences"].iloc[0])
        print("--------------------------------")

        # Print parsing statistics
        successful_parses = results_df["parse_error"].isna().sum()
        total_responses = len(results_df)
        final_parsing_rate = successful_parses/total_responses*100 if total_responses > 0 else 0
        final_truncation_rate = truncated_count/total_responses*100 if total_responses > 0 else 0
        print(f"Final JSON parsing success rate: {successful_parses}/{total_responses} ({final_parsing_rate:.1f}%)")
        print(f"Final truncation rate: {truncated_count}/{total_responses} ({final_truncation_rate:.1f}%)")

        # Save the final dataframe
        results_df.to_json(args.output_file.replace(".csv", ".jsonl"), orient='records', lines=True)
        print(f"Final results saved to {args.output_file}")
        print(f"Dataframe shape: {results_df.shape}")
        print(f"Columns: {list(results_df.columns)}")

        # Cast all columns to string for wandb
        results_df_str = results_df.astype(str)

        # Final wandb log with complete results
        run.log({"final_results": wandb.Table(dataframe=results_df_str)})

        # Log final summary to wandb
        wandb.log({
            "final_parsing_success_rate": final_parsing_rate,
            "final_total_processed": total_responses,
            "final_successful_parses": successful_parses,
            "final_battles_processed": processed_count,
            "final_truncation_rate": final_truncation_rate,
            "final_truncated_battles": truncated_count
        })
    else:
        print("No valid battles were processed!")

    wandb.finish()

if __name__ == "__main__":
    main() 