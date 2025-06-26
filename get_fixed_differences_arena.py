import re
import argparse
import os
import json
from vllm import LLM, SamplingParams
import pandas as pd
import wandb
from transformers import AutoTokenizer
from collections import Counter
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Try to import litellm for GPT models
try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm not available. GPT models will not be supported.")

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

def format_arena_example(user_prompt, model_a_name, model_a_response, model_b_name, model_b_response, reverse_order=False):
    """Format the arena data for the model to compare."""
    if reverse_order:
        return f"# Prompt: {user_prompt}\n\n# Model A response: {model_b_response}\n\n--------------------------------\n\n# Model B response: {model_a_response}"
    else:
        return f"# Prompt: {user_prompt}\n\n# Model A response: {model_a_response}\n\n--------------------------------\n\n# Model B response: {model_b_response}"

def remove_thinking_from_output(output):
    """Remove thinking sections from model output."""
    # Handle NaN or non-string values
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

def get_token_count(user_prompt, model_a_response, model_b_response, tokenizer, system_prompt):
    """Helper function to get token count for a formatted message."""
    formatted = format_arena_example(user_prompt, "ModelA", model_a_response, "ModelB", model_b_response)
    message = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": formatted}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tokenizer.encode(message))

def truncate_long_responses(user_prompt, model_a_response, model_b_response, tokenizer, max_model_len, system_prompt, safety_margin=500):
    """
    Truncate model responses if the total prompt would exceed max_model_len.
    Preserves the user prompt and system prompt, truncating model responses proportionally.
    Returns tuple of (truncated_a_response, truncated_b_response, was_truncated)
    """
    # Check if truncation is needed
    current_length = get_token_count(user_prompt, model_a_response, model_b_response, tokenizer, system_prompt)
    
    if current_length <= max_model_len:
        return model_a_response, model_b_response, False
    
    print(f"Prompt too long ({current_length} tokens > {max_model_len}). Truncating responses...")
    
    # Calculate base overhead tokens (everything except the model responses)
    base_tokens = get_token_count(user_prompt, "", "", tokenizer, system_prompt)
    
    # Calculate available tokens for responses
    available_tokens = max_model_len - base_tokens - safety_margin
    
    if available_tokens < 100:  # Minimum total response length
        print(f"Warning: Very little space for responses ({available_tokens} tokens total)")
        available_tokens = 100
    
    # Split available tokens between the two responses and estimate character limit
    tokens_per_response = available_tokens // 2
    max_chars_per_response = tokens_per_response * 3  # Rough estimate: ~3 chars per token
    
    # Initial character-based truncation
    truncated_a_response = model_a_response[:max_chars_per_response] + " [TRUNCATED]" if len(model_a_response) > max_chars_per_response else model_a_response
    truncated_b_response = model_b_response[:max_chars_per_response] + " [TRUNCATED]" if len(model_b_response) > max_chars_per_response else model_b_response
    
    # Token-based fine-tuning with iterative reduction
    max_iterations = 3
    reduction_factor = 0.75
    
    for iteration in range(max_iterations):
        current_length = get_token_count(user_prompt, truncated_a_response, truncated_b_response, tokenizer, system_prompt)
        
        print(f"Iteration {iteration + 1}: Current message length = {current_length} tokens")
        
        if current_length <= max_model_len:
            print(f"Successfully truncated to {current_length} tokens")
            return truncated_a_response, truncated_b_response, True
        
        # Reduce both responses by the reduction factor
        current_a_length = len(truncated_a_response)
        current_b_length = len(truncated_b_response)
        
        new_a_length = int(current_a_length * reduction_factor)
        new_b_length = int(current_b_length * reduction_factor)
        
        truncated_a_response = truncated_a_response[:new_a_length] + " [TRUNCATED]"
        truncated_b_response = truncated_b_response[:new_b_length] + " [TRUNCATED]"
        
        print(f"Reduced response lengths to {new_a_length} and {new_b_length} characters")
    
    # Final check with emergency truncation if needed
    final_length = get_token_count(user_prompt, truncated_a_response, truncated_b_response, tokenizer, system_prompt)
    
    if final_length > max_model_len:
        print(f"Warning: Could not truncate to fit within {max_model_len} tokens after {max_iterations} iterations (final length: {final_length})")
        # Emergency truncation - cut to very short responses
        emergency_length = 200
        truncated_a_response = model_a_response[:emergency_length] + " [HEAVILY TRUNCATED]"
        truncated_b_response = model_b_response[:emergency_length] + " [HEAVILY TRUNCATED]"
    
    return truncated_a_response, truncated_b_response, True

def analyze_comparison_proportions(results_df):
    """Analyze the proportions of A/B/Tie outcomes across all axes and samples."""
    all_outcomes = []
    
    # Extract all comparison outcomes
    for _, row in results_df.iterrows():
        if pd.isna(row['comparisons']) or row['comparisons'] == 'None' or row['comparisons'] is None:
            continue
            
        try:
            comparisons = row['comparisons']
            
            # Handle string representation of dict
            if isinstance(comparisons, str):
                try:
                    comparisons = json.loads(comparisons.replace("'", '"'))
                except json.JSONDecodeError:
                    continue
            
            # Make sure comparisons is a dictionary
            if isinstance(comparisons, dict):
                # Extract only string values (A, B, Tie) and skip any non-string values
                for value in comparisons.values():
                    if isinstance(value, str) and value in ['A', 'B', 'Tie']:
                        all_outcomes.append(value)
                        
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            print(f"Warning: Error parsing comparisons data: {e}")
            continue
    
    # Count outcomes
    outcome_counts = Counter(all_outcomes)
    total_outcomes = sum(outcome_counts.values())
    
    if total_outcomes == 0:
        return {}, {}
    
    # Calculate proportions
    proportions = {
        outcome: count / total_outcomes 
        for outcome, count in outcome_counts.items()
    }
    
    return outcome_counts, proportions

def save_intermediate_results(results_df, output_file, batch_num):
    """Save intermediate results to the main output file and log to wandb."""
    # Save to the main output file (overwrite with updated results)
    results_df.to_csv(output_file, index=False)
    print(f"Updated results saved to {output_file} (after batch {batch_num})")
    
    # Analyze comparison proportions
    outcome_counts, proportions = analyze_comparison_proportions(results_df)
    
    # Print proportions
    if proportions:
        print(f"\nComparison outcome proportions (after batch {batch_num}):")
        for outcome, prop in sorted(proportions.items()):
            count = outcome_counts[outcome]
            print(f"  {outcome}: {count} ({prop:.1%})")
        print(f"  Total outcomes: {sum(outcome_counts.values())}")
    
    # Log to wandb - both incremental and cumulative views
    batch_size = 10  # Assuming 10 per batch
    start_idx = max(0, (batch_num - 1) * batch_size)
    end_idx = min(len(results_df), batch_num * batch_size)
    
    # Log just this batch's results
    current_batch_results = results_df.iloc[start_idx:end_idx]
    
    # Log both the current batch and cumulative results
    log_data = {
        "batch_completed": batch_num,
        "total_processed": len(results_df),
        f"batch_{batch_num}_results": wandb.Table(dataframe=current_batch_results.astype(str)),  # This batch only
        "all_results_so_far": wandb.Table(dataframe=results_df.astype(str))  # All results accumulated
    }
    
    # Add proportion metrics
    if proportions:
        for outcome, prop in proportions.items():
            log_data[f"proportion_{outcome}"] = prop
            log_data[f"count_{outcome}"] = outcome_counts[outcome]
        log_data["total_comparison_outcomes"] = sum(outcome_counts.values())
    
    wandb.log(log_data)

def generate_with_litellm_single(message, model_name, temperature=0.7, max_tokens=4096):
    """Generate a single response using LiteLLM for GPT models."""
    try:
        # Convert the formatted message to chat format
        # Extract the system and user content from the formatted message
        if "### System:" in message:
            parts = message.split("### System:")
            if len(parts) > 1:
                system_part = parts[1].split("### User:")[0].strip()
                user_part = parts[1].split("### User:")[1].strip() if "### User:" in parts[1] else ""
            else:
                system_part = ""
                user_part = message
        else:
            system_part = labeling_systems_prompt
            user_part = message
        
        # Create messages for LiteLLM
        chat_messages = [
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_part}
        ]
        
        response = completion(
            model=model_name,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating with LiteLLM: {e}")
        return f"Error: {str(e)}"

def generate_with_litellm(messages, model_name, temperature=0.7, max_tokens=4096, max_workers=10):
    """Generate responses using LiteLLM for GPT models with threading for speed."""
    if not LITELLM_AVAILABLE:
        raise ImportError("litellm is required for GPT models. Install with: pip install litellm")
    
    print(f"Generating {len(messages)} responses with {max_workers} threads...")
    start_time = time.time()
    
    responses = [None] * len(messages)  # Initialize responses list with correct size
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(generate_with_litellm_single, message, model_name, temperature, max_tokens): i
            for i, message in enumerate(messages)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                responses[index] = result
                completed += 1
                if completed % 5 == 0 or completed == len(messages):
                    print(f"Completed {completed}/{len(messages)} API calls...")
            except Exception as e:
                print(f"Error in thread for message {index}: {e}")
                responses[index] = f"Thread Error: {str(e)}"
    
    elapsed_time = time.time() - start_time
    print(f"Generated {len(messages)} responses in {elapsed_time:.2f} seconds ({len(messages)/elapsed_time:.2f} req/sec)")
    
    return responses

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare model responses using fixed axes on arena data')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of battles to process (default: 100)')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-32B",
                       help='Model name to use for the LLM')
    parser.add_argument('--output_file', type=str, default="disguising/misc/arena_model_comparison_fixed_axes.csv",
                       help='Output file to save the results')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top p')
    parser.add_argument('--max_tokens', type=int, default=4096,
                       help='Max tokens')
    parser.add_argument('--max_model_len', type=int, default=16384,
                       help='Max model length')
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
    parser.add_argument('--max_threads', type=int, default=16,
                       help='Maximum number of threads for LiteLLM API calls (default: 10)')
    args = parser.parse_args()

    run = wandb.init(project="arena-fixed-axes", name="arena_fixed_axes_comparison")
    run.summary["system_prompt"] = labeling_systems_prompt
    run.config.update(vars(args))
    run.config["system_prompt_type"] = "fixed_axes_arena"

    # make output directory if it doesn't exist
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
    
    if args.min_turn:
        df = df[df['turn'] >= args.min_turn]
        print(f"After min_turn filter: {len(df)} battles")

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

    # If running both sides, we'll process each battle twice
    if args.run_both_sides:
        print(f"Running both sides - will process {len(df) * 2} total comparisons")

    # Initialize the appropriate LLM based on model name
    use_litellm = args.model_name.lower().startswith('gpt')
    
    if use_litellm:
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for GPT models. Install with: pip install litellm")
        print(f"Using LiteLLM for model: {args.model_name}")
        # For LiteLLM, we don't need to initialize an LLM object
        llm = None
        tokenizer = None  # We'll use a simple approximation for token counting
    else:
        print(f"Using vLLM for model: {args.model_name}")
        llm = LLM(
            model=args.model_name,
            tokenizer=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Process battles in batches for periodic saving
    new_data = {
        "question_id": [], 
        "user_prompt": [], 
        "model_1_response": [], 
        "model_2_response": [], 
        "model_1_name": [], 
        "model_2_name": [], 
        "model_order": [],  # New field to track A-B vs B-A ordering
        "winner": [],
        "raw_response": [],  # Raw model output before parsing
        "comparisons": [], 
        "rationales": [], 
        "parse_error": []
    }

    batch_size = 100
    processed_count = 0
    truncated_count = 0
    
    # Create list of all comparisons to run
    all_comparisons = []
    for _, row in df.iterrows():
        all_comparisons.append((row, False))  # Normal order (A=model_a, B=model_b)
        if args.run_both_sides:
            all_comparisons.append((row, True))   # Reversed order (A=model_b, B=model_a)
    
    for batch_start in range(0, len(all_comparisons), batch_size):
        batch_end = min(batch_start + batch_size, len(all_comparisons))
        batch_comparisons = all_comparisons[batch_start:batch_end]
        batch_messages = []
        batch_data = {
            "question_id": [],
            "user_prompt": [], 
            "model_1_response": [], 
            "model_2_response": [], 
            "model_1_name": [], 
            "model_2_name": [], 
            "model_order": [],
            "winner": []
        }
        
        print(f"Processing batch {batch_start//batch_size + 1}: comparisons {batch_start+1}-{batch_end}")
        
        for row, reverse_order in batch_comparisons:
            try:
                # Extract conversation data
                user_prompt_a, model_a_response = extract_content_from_conversation(row['conversation_a'])
                user_prompt_b, model_b_response = extract_content_from_conversation(row['conversation_b'])
                
                # Skip if we can't extract proper conversation data
                if not user_prompt_a or not model_a_response or not user_prompt_b or not model_b_response:
                    continue
                    
                # The user prompts should be the same, use the first one
                user_prompt = user_prompt_a
                model_a_name = row['model_a']
                model_b_name = row['model_b']
                
                # Truncate responses if necessary
                if args.auto_truncate:
                    truncated_a_response, truncated_b_response, was_truncated = truncate_long_responses(
                        user_prompt, model_a_response, model_b_response, tokenizer, args.max_model_len, labeling_systems_prompt, args.truncation_safety_margin
                    )
                    if was_truncated:
                        truncated_count += 1
                else:
                    truncated_a_response, truncated_b_response = model_a_response, model_b_response
                
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
                
                message = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": labeling_systems_prompt}, 
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
                batch_data["model_1_response"].append(model_1_response)
                batch_data["model_2_response"].append(model_2_response)
                batch_data["model_1_name"].append(model_1_name)
                batch_data["model_2_name"].append(model_2_name)
                batch_data["model_order"].append(model_order)
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
            if use_litellm:
                # For GPT models, use a simple approximation (4 chars = 1 token roughly)
                token_count = len(message) // 4
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
        if use_litellm:
            batch_responses_text = generate_with_litellm(
                valid_messages, 
                args.model_name, 
                temperature=args.temperature, 
                max_tokens=args.max_tokens,
                max_workers=args.max_threads
            )
            # Convert to objects that mimic vLLM response structure
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            
            class MockOutput:
                def __init__(self, text):
                    self.outputs = [MockResponse(text)]
            
            batch_responses = [MockOutput(text) for text in batch_responses_text]
        else:
            batch_responses = llm.generate(valid_messages, sampling_params=sampling_params)
        
        # Process the responses (only for valid indices)
        for response_idx, response in enumerate(batch_responses):
            original_idx = valid_indices[response_idx]  # Map back to original index
            if original_idx >= len(batch_data["question_id"]):  # Safety check
                break
                
            cleaned_response = remove_thinking_from_output(response.outputs[0].text)
            parsed_json, parse_error = parse_json_response(cleaned_response)
            
            # Add to main data structure
            new_data["question_id"].append(batch_data["question_id"][original_idx])
            new_data["user_prompt"].append(batch_data["user_prompt"][original_idx])
            new_data["model_1_response"].append(batch_data["model_1_response"][original_idx])
            new_data["model_2_response"].append(batch_data["model_2_response"][original_idx])
            new_data["model_1_name"].append(batch_data["model_1_name"][original_idx])
            new_data["model_2_name"].append(batch_data["model_2_name"][original_idx])
            new_data["model_order"].append(batch_data["model_order"][original_idx])
            new_data["winner"].append(batch_data["winner"][original_idx])
            new_data["raw_response"].append(cleaned_response)
            new_data["comparisons"].append(parsed_json.get("comparisons") if parsed_json else None)
            new_data["rationales"].append(parsed_json.get("rationales") if parsed_json else None)
            new_data["parse_error"].append(parse_error)
            
            processed_count += 1
        
        # Create current results dataframe and save after each batch
        current_results_df = pd.DataFrame(new_data).dropna(subset=['question_id'])
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
        print(results_df["comparisons"].iloc[0])
        print("\nSample rationales:")
        print(results_df["rationales"].iloc[0])
        print("--------------------------------")

        # Analyze final comparison proportions
        final_outcome_counts, final_proportions = analyze_comparison_proportions(results_df)
        
        # Print final proportions
        if final_proportions:
            print(f"\nFinal comparison outcome proportions:")
            for outcome, prop in sorted(final_proportions.items()):
                count = final_outcome_counts[outcome]
                print(f"  {outcome}: {count} ({prop:.1%})")
            print(f"  Total outcomes: {sum(final_outcome_counts.values())}")

        # If running both sides, also analyze by ordering
        if args.run_both_sides:
            print(f"\nAnalysis by model ordering:")
            for order in ["A-B", "B-A"]:
                order_df = results_df[results_df["model_order"] == order]
                if len(order_df) > 0:
                    order_counts, order_proportions = analyze_comparison_proportions(order_df)
                    if order_proportions:
                        print(f"  {order} ordering:")
                        for outcome, prop in sorted(order_proportions.items()):
                            count = order_counts[outcome]
                            print(f"    {outcome}: {count} ({prop:.1%})")

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

        # cast all columns to string
        results_df = results_df.astype(str)

        # Final wandb log with complete results
        final_log_data = {"final_results": wandb.Table(dataframe=results_df)}
        
        # Add final proportion metrics
        if final_proportions:
            for outcome, prop in final_proportions.items():
                final_log_data[f"final_proportion_{outcome}"] = prop
                final_log_data[f"final_count_{outcome}"] = final_outcome_counts[outcome]
            final_log_data["final_total_comparison_outcomes"] = sum(final_outcome_counts.values())
        
        run.log(final_log_data)

        # Log final summary to wandb
        wandb_summary = {
            "final_parsing_success_rate": final_parsing_rate,
            "final_total_processed": total_responses,
            "final_successful_parses": successful_parses,
            "final_battles_processed": processed_count,
            "final_truncation_rate": final_truncation_rate,
            "final_truncated_battles": truncated_count
        }
        
        # Add ordering-specific metrics if running both sides
        if args.run_both_sides:
            for order in ["A-B", "B-A"]:
                order_df = results_df[results_df["model_order"] == order]
                order_counts, order_proportions = analyze_comparison_proportions(order_df)
                if order_proportions:
                    for outcome, prop in order_proportions.items():
                        wandb_summary[f"{order}_proportion_{outcome}"] = prop
                        wandb_summary[f"{order}_count_{outcome}"] = order_counts[outcome]
        
        wandb.log(wandb_summary)
    else:
        print("No valid battles were processed!")

    wandb.finish()

if __name__ == "__main__":
    main()