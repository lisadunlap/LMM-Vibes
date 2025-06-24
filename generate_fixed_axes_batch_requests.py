import re
import argparse
import os
import json
import pandas as pd
import wandb
from transformers import AutoTokenizer
from datasets import load_dataset

labeling_systems_prompt = """You are an expert judge comparing two responses (A and B) to the same user prompt. For each axis defined below, decide which response better exhibits that property. Use only the definitions given. If neither clearly wins or neither response has the property, return "Tie". 


Here are the definitions of the axes:
friendliness: warmth or approachability in tone.
formality: level of formality vs. casualness.
politeness: courtesy without undue flattery.
sycophancy: excessive or insincere agreement/flattery.
empathy: acknowledgement of user feelings or perspective.
humor: light-hearted or witty elements.
anthropomorphism: attributing human-like qualities to non-human entities.
assertiveness: confidence vs. hedging.
directness: straightforwardness and unambiguity.
conciseness: succinctness; avoid unnecessary detail.
specificity: presence of concrete details vs. vagueness.
creativity: originality or novel perspective.
depth: thoroughness in addressing the prompt.
relevance: focus on the prompt without drift.
context_awareness: appropriate use of prior context.
safety: refraining from harmful or risky suggestions.
refusal_to_answer: refusal to answer the prompt.
ethical_sensitivity: awareness of ethical implications.
actionability: presence of concrete next steps or advice.
user_intent_alignment: how well it matches user's need and tone.
helpfulness: overall usefulness for the user's goal.
engagement: invites further interaction or follow-up.
transparency: clarity about uncertainties or limitations.
gen_z: use of slang, emojis, and other modern language features.

Output a JSON object with this structure (select either A B or Tie for each axis):
{
  "comparisons": {
    {{axis_name}}: [choice - A, B, or Tie],
    {{axis_name}}: [choice - A, B, or Tie],
    ...
  }
}

Do not include any other keys. Use exactly the axis names as keys.
"""

def extract_content_from_conversation(conversation):
    """Extract the user prompt and assistant response from conversation format."""
    return conversation[0]['content'], conversation[1]['content']

def format_arena_example(user_prompt, model_a_name, model_a_response, model_b_name, model_b_response, reverse_order=False):
    """Format the arena data for the model to compare."""
    if reverse_order:
        return f"# Prompt: {user_prompt}\n\n# Model A response: {model_b_response}\n\n--------------------------------\n\n# Model B response: {model_a_response}"
    else:
        return f"# Prompt: {user_prompt}\n\n# Model A response: {model_a_response}\n\n--------------------------------\n\n# Model B response: {model_b_response}"

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
    parser = argparse.ArgumentParser(description='Generate OpenAI batch API requests for fixed axes arena model comparison')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of battles to process')
    parser.add_argument('--model_name', type=str, default="gpt-4.1-2025-04-14",
                       help='OpenAI model name to use for the batch requests')
    parser.add_argument('--output_file', type=str, default="arena_fixed_axes_batch_requests.jsonl",
                       help='Output file to save the batch requests')
    parser.add_argument('--metadata_file', type=str, default="arena_fixed_axes_batch_metadata.json",
                       help='Output file to save the metadata')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top p')
    parser.add_argument('--max_tokens', type=int, default=4096,
                       help='Max tokens')
    parser.add_argument('--max_model_len', type=int, default=16384,
                       help='Max model length for truncation')
    parser.add_argument('--run_both_sides', action='store_true',
                       help='Run each comparison twice with different model orderings to reduce position bias')
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
    run = wandb.init(project="arena-fixed-axes-batch-generation", name="arena_fixed_axes_batch_gen")
    run.summary["system_prompt"] = labeling_systems_prompt
    run.config.update(vars(args))
    run.config["system_prompt_type"] = "fixed_axes_arena"

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

    # If running both sides, we'll process each battle twice
    if args.run_both_sides:
        print(f"Running both sides - will process {len(df) * 2} total comparisons")

    # Initialize tokenizer for truncation estimation
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer as approximation

    # Process battles and generate batch requests
    batch_requests = []
    batch_metadata = []  # Store metadata separately
    processed_count = 0
    truncated_count = 0
    skipped_count = 0
    
    # Create list of all comparisons to run
    all_comparisons = []
    for _, row in df.iterrows():
        all_comparisons.append((row, False))  # Normal order (A=model_a, B=model_b)
        if args.run_both_sides:
            all_comparisons.append((row, True))   # Reversed order (A=model_b, B=model_a)
    
    print(f"Processing {len(all_comparisons)} comparisons to generate batch requests...")
    
    for comparison_idx, (row, reverse_order) in enumerate(all_comparisons):
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
                    args.max_model_len, labeling_systems_prompt, args.truncation_safety_margin
                )
                if was_truncated:
                    truncated_count += 1
            else:
                truncated_a_response, truncated_b_response = model_a_response, model_b_response
                was_truncated = False
            
            # Determine the actual model names and responses based on ordering
            if reverse_order:
                # When reversed: A=model_b, B=model_a in the prompt, but we store original model_a/model_b
                model_1_name = model_a_name  # Still store as model_1
                model_2_name = model_b_name  # Still store as model_2
                model_1_response = truncated_a_response
                model_2_response = truncated_b_response
                model_order = "B-A"  # B first in prompt, A second
            else:
                model_1_name = model_a_name
                model_2_name = model_b_name
                model_1_response = truncated_a_response
                model_2_response = truncated_b_response
                model_order = "A-B"  # A first in prompt, B second
            
            # Create the formatted input for the LLM
            formatted_input = format_arena_example(
                user_prompt, model_a_name, truncated_a_response, model_b_name, truncated_b_response, reverse_order
            )
            
            # Create unique custom_id for batch request
            custom_id = f"arena-fixed-axes-{row['question_id']}"
            if reverse_order:
                custom_id += "-reversed"
            
            # Create the batch request in OpenAI format
            batch_request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model_name,
                    "messages": [
                        {"role": "system", "content": labeling_systems_prompt},
                        {"role": "user", "content": formatted_input}
                    ],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens
                }
            }
            
            batch_requests.append(batch_request)
            batch_metadata.append({
                'custom_id': custom_id,
                'question_id': row['question_id'],
                'user_prompt': user_prompt,
                'model_1_name': model_1_name,
                'model_2_name': model_2_name,
                'model_1_response': model_1_response,
                'model_2_response': model_2_response,
                'model_order': model_order,
                'winner': row['winner'],
                'was_truncated': was_truncated
            })
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} comparisons...")
                
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
    print(f"Total comparisons processed: {processed_count}")
    print(f"Comparisons skipped: {skipped_count}")
    print(f"Truncation rate: {truncated_count}/{total_processed} ({truncation_rate:.1f}%)")
    print(f"Skip rate: {skipped_count}/{total_processed + skipped_count} ({skip_rate:.1f}%)")
    print(f"Batch requests generated: {len(batch_requests)}")
    
    if args.run_both_sides:
        print(f"Running both sides: Each of {len(df)} battles generated 2 comparisons")
    
    # Log to wandb
    wandb.log({
        "total_comparisons_processed": processed_count,
        "comparisons_skipped": skipped_count,
        "truncation_rate": truncation_rate,
        "truncated_comparisons": truncated_count,
        "skip_rate": skip_rate,
        "batch_requests_generated": len(batch_requests),
        "run_both_sides": args.run_both_sides,
        "unique_battles": len(df)
    })
    
    wandb.finish()
    
    print(f"\nTo submit to OpenAI Batch API:")
    print(f"1. Upload the file: {args.output_file}")
    print(f"2. Keep the metadata file for processing results: {args.metadata_file}")
    print(f"3. Use the metadata file to match results back to original data when processing batch results")

if __name__ == "__main__":
    main() 