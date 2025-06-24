import re
import argparse
import os
import json
import pandas as pd
import wandb
from transformers import AutoTokenizer
from datasets import load_dataset

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
Produce a JSON list of objects. Each object will represent a single distinct property observed in one model's response that is notably absent or different in the other's. Focus on identifying key areas of distinction, and the individual property observations in the output list (e.g., Model A's formal tone would be one entry, Model B's casual tone would be another related entry). As these are very common and easy to measure with heuristics, please do not include properties like "Model A is more concise than Model B". If applicatble, make sure to include properties revolving around the models reasoning, interpretation of the prompt/intent, and potential reason for errors if they exist. 

**Fields to include:**
*   **Model:** Which model has this property?
*   **Property:** A single distinct property observed in one model's response that is notably absent or different in the other's.
    *   *Think:* Could this property affect the user's experience? Does this property possibly contribute to a high or low quality response? Could this be a property that is seen in responses to other prompts?
*   **Category:** A 1-4 word category that describes the property.
    *   *Think:* What category does this property fall into? How would someone group this property?
*   **General Trait:** Reflects a model's typical behavior across diverse prompts.
    *   *Think:* Is this how this Model *usually* is compared to the other?
*   **Reason:** A brief justification for this property, noting its absence/difference in the other model.
    *   *Think:* Why is this property present in one model but not the other?
*   **Context-Specific Difference:** Arises mainly due to *this specific* user prompt.
    *   *Think:* Is this property a direct reaction to *this current prompt*?
*   **Impact:** How much does this property impact the user's experience?
    *   *Think:* How notable is this difference? Would this be a property that a user would notice and favor or not favor? Note that this could depend on the user's intent and the context of the prompt.
*   **Contains Errors:** Does either model response contain errors?
    *   *Think:* Are there factual errors, hallucinations, or other strange or unwanted behavior?
*   **Unexpected Behavior:** Does the model's response contain highly unusual or concerning behavior? If true then a developer will analyze these responses manually.
    *   *Think:* Does this involve offensive language, gibberish, bias, factual hallucinations, or other strange behavior?

**JSON Output Structure for each property (BE BRIEF, if no notable properties exist, return empty list. Please use the names of the models in the output rather than "Model A"/"Model B"):**
```json
[
  {
    "model": "Model A|Model B",
    "property_description": "Brief description of the unique property observed in this model (max 2 sentences)",
    "category": "1-4 word category",
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

def truncate_long_responses(user_prompt, model_a_response, model_b_response, tokenizer, max_model_len, system_prompt, safety_margin=500):
    """
    Truncate model responses if the total prompt would exceed max_model_len.
    Preserves the user prompt and system prompt, truncating model responses proportionally.
    Returns tuple of (truncated_a_response, truncated_b_response, was_truncated)
    """
    
    def estimate_token_count(text):
        """Estimate token count for OpenAI (roughly 4 chars per token)."""
        return len(text) // 4
    
    def create_full_message(user_prompt, model_a_response, model_b_response, system_prompt):
        """Create the full message as it would be sent to OpenAI."""
        formatted_input = format_arena_example(
            user_prompt, "ModelA", model_a_response, "ModelB", model_b_response
        )
        return system_prompt + "\n\n" + formatted_input
    
    # Create a test message to estimate token count
    test_message = create_full_message(
        user_prompt, model_a_response, model_b_response, system_prompt
    )
    current_length = estimate_token_count(test_message)
    
    # If within limit, return original responses
    if current_length <= max_model_len:
        return model_a_response, model_b_response, False
    
    print(f"Prompt too long ({current_length} tokens > {max_model_len}). Truncating responses...")
    
    # Calculate base overhead tokens (everything except the model responses)
    base_message = create_full_message(
        user_prompt, "", "", system_prompt
    )
    base_tokens = estimate_token_count(base_message)
    
    # Calculate available tokens for responses
    available_tokens = max_model_len - base_tokens - safety_margin
    
    if available_tokens < 100:  # Minimum total response length
        print(f"Warning: Very little space for responses ({available_tokens} tokens total)")
        available_tokens = 100
    
    # Split available tokens between the two responses
    tokens_per_response = available_tokens // 2
    
    # Convert tokens to approximate characters (~4 chars per token for OpenAI)
    max_chars_per_response = tokens_per_response * 4
    
    # Initial truncation by characters
    truncated_a_response = model_a_response[:max_chars_per_response] + " [TRUNCATED]" if len(model_a_response) > max_chars_per_response else model_a_response
    truncated_b_response = model_b_response[:max_chars_per_response] + " [TRUNCATED]" if len(model_b_response) > max_chars_per_response else model_b_response
    
    # Fine-tune with iterative approach
    max_iterations = 3
    
    for iteration in range(max_iterations):
        test_message = create_full_message(
            user_prompt, truncated_a_response, truncated_b_response, system_prompt
        )
        current_length = estimate_token_count(test_message)
        
        print(f"Iteration {iteration + 1}: Current message length = {current_length} tokens")
        
        if current_length <= max_model_len:
            print(f"Successfully truncated to {current_length} tokens")
            return truncated_a_response, truncated_b_response, True
        
        # If still too long, cut both responses by 25%
        reduction_factor = 0.75  # Keep 75% of current length
        
        current_a_length = len(truncated_a_response)
        current_b_length = len(truncated_b_response)
        
        new_a_length = int(current_a_length * reduction_factor)
        new_b_length = int(current_b_length * reduction_factor)
        
        truncated_a_response = truncated_a_response[:new_a_length] + " [TRUNCATED]"
        truncated_b_response = truncated_b_response[:new_b_length] + " [TRUNCATED]"
        
        print(f"Reduced response lengths to {new_a_length} and {new_b_length} characters")
    
    # Final check - if still too long, emergency truncation
    test_message = create_full_message(
        user_prompt, truncated_a_response, truncated_b_response, system_prompt
    )
    final_length = estimate_token_count(test_message)
    
    if final_length > max_model_len:
        print(f"Warning: Could not truncate to fit within {max_model_len} tokens after {max_iterations} iterations (final length: {final_length})")
        # Emergency truncation - cut to very short responses
        emergency_length = 200  # Very short responses
        truncated_a_response = model_a_response[:emergency_length] + " [HEAVILY TRUNCATED]"
        truncated_b_response = model_b_response[:emergency_length] + " [HEAVILY TRUNCATED]"
    
    return truncated_a_response, truncated_b_response, True

def save_batch_requests(requests, metadata_list, output_file, metadata_file):
    """Save batch requests to JSONL file and metadata to separate file."""
    # Save requests
    with open(output_file, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"Saved {len(requests)} batch requests to {output_file}")
    print(f"Saved metadata to {metadata_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate OpenAI batch API requests for arena dataset model comparison')
    parser.add_argument('--num_samples', type=int,
                       help='Number of battles to process')
    parser.add_argument('--model_name', type=str, default="gpt-4.1-2025-04-14",
                       help='OpenAI model name to use for the batch requests')
    parser.add_argument('--output_file', type=str, default="arena_batch_requests.jsonl",
                       help='Output file to save the batch requests')
    parser.add_argument('--metadata_file', type=str, default="arena_batch_metadata.json",
                       help='Output file to save the metadata')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top p')
    parser.add_argument('--max_tokens', type=int, default=2048,
                       help='Max tokens')
    parser.add_argument('--max_model_len', type=int, default=16384,
                       help='Max model length for truncation')
    parser.add_argument('--filter_english', action='store_true', default=True,
                       help='Filter to only English conversations')
    parser.add_argument('--min_turn', type=int, default=1,
                       help='Minimum number of turns in conversation')
    parser.add_argument('--exclude_ties', action='store_true', default=True,
                       help='Exclude tied battles')
    parser.add_argument('--ties_only', action='store_true', default=False,
                       help='Only include tied battles')
    parser.add_argument('--auto_truncate', action='store_true', default=True,
                       help='Automatically truncate long prompts to fit within max_model_len')
    parser.add_argument('--truncation_safety_margin', type=int, default=1000,
                       help='Reserve this many tokens as safety margin when truncating (default: 1000)')
    args = parser.parse_args()

    # Initialize wandb for tracking
    run = wandb.init(project="arena-batch-request-generation", name="arena_batch_gen")
    run.summary["system_prompt"] = one_sided_system_prompt
    run.config.update(vars(args))
    run.config["system_prompt_type"] = "one_sided_json_arena"

    # Make output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)

    # Load the arena dataset
    print("Loading arena dataset...")
    dataset = load_dataset("lmarena-ai/arena-human-preference-100k", split="train")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} battles from arena dataset")

    # Apply filters
    if args.filter_english:
        df = df[df['language'] == 'English']
        print(f"After English filter: {len(df)} battles")

    if args.min_turn:
        df = df[df['turn'] >= args.min_turn]
        print(f"After min_turn filter: {len(df)} battles")
    
    models = [
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

    # Initialize tokenizer for truncation estimation
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer as approximation

    # Process battles and generate batch requests
    batch_requests = []
    batch_metadata = []  # Store metadata separately
    processed_count = 0
    truncated_count = 0
    skipped_count = 0
    
    print(f"Processing {len(df)} battles to generate batch requests...")
    
    for idx, row in df.iterrows():
        try:
            # Extract conversation data
            user_prompt_a, model_a_response = extract_content_from_conversation(row['conversation_a'])
            user_prompt_b, model_b_response = extract_content_from_conversation(row['conversation_b'])
            
            # Skip if we can't extract proper conversation data
            if not user_prompt_a or not model_a_response or not user_prompt_b or not model_b_response:
                skipped_count += 1
                continue
                
            # The user prompts should be the same, use the first one
            user_prompt = user_prompt_a
            model_a_name = row['model_a']
            model_b_name = row['model_b']
            
            # Truncate responses if necessary
            if args.auto_truncate:
                truncated_a_response, truncated_b_response, was_truncated = truncate_long_responses(
                    user_prompt, model_a_response, model_b_response, tokenizer, 
                    args.max_model_len, one_sided_system_prompt, args.truncation_safety_margin
                )
                if was_truncated:
                    truncated_count += 1
            else:
                truncated_a_response, truncated_b_response = model_a_response, model_b_response
            
            # Create the formatted input for the LLM
            formatted_input = format_arena_example(
                user_prompt, model_a_name, truncated_a_response, model_b_name, truncated_b_response
            )
            
            # Create the batch request in OpenAI format
            batch_request = {
                "custom_id": f"arena-comparison-{row['question_id']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model_name,
                    "messages": [
                        {"role": "system", "content": one_sided_system_prompt},
                        {"role": "user", "content": formatted_input}
                    ],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens
                }
            }
            
            batch_requests.append(batch_request)
            batch_metadata.append({
                "custom_id": f"arena-comparison-{row['question_id']}",
                "question_id": row['question_id'],
                "prompt": user_prompt,
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
                "model_a_response": model_a_response,
                "model_b_response": model_b_response,
                "winner": row['winner'],
                "was_truncated": was_truncated if args.auto_truncate else False
            })
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} battles...")
                
        except Exception as e:
            print(f"Error processing row with question_id {row.get('question_id', 'unknown')}: {str(e)}")
            skipped_count += 1
            continue

    # Save the batch requests
    save_batch_requests(batch_requests, batch_metadata, args.output_file, args.metadata_file)
    
    # Print final statistics
    total_processed = processed_count
    truncation_rate = truncated_count/total_processed*100 if total_processed > 0 else 0
    skip_rate = skipped_count/(total_processed + skipped_count)*100 if (total_processed + skipped_count) > 0 else 0
    
    print(f"\nFinal Statistics:")
    print(f"Total battles processed: {processed_count}")
    print(f"Battles skipped: {skipped_count}")
    print(f"Truncation rate: {truncated_count}/{total_processed} ({truncation_rate:.1f}%)")
    print(f"Skip rate: {skipped_count}/{total_processed + skipped_count} ({skip_rate:.1f}%)")
    print(f"Batch requests generated: {len(batch_requests)}")
    
    # Log to wandb
    wandb.log({
        "total_battles_processed": processed_count,
        "battles_skipped": skipped_count,
        "truncation_rate": truncation_rate,
        "truncated_battles": truncated_count,
        "skip_rate": skip_rate,
        "batch_requests_generated": len(batch_requests)
    })
    
    wandb.finish()
    
    print(f"\nTo submit to OpenAI Batch API:")
    print(f"1. Upload the file: {args.output_file}")
    print(f"2. Keep the metadata file for processing results: {args.metadata_file}")
    print(f"3. Use the metadata file to match results back to original data when processing batch results")

if __name__ == "__main__":
    main() 