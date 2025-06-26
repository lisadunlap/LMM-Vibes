import re
import argparse
import os
import json
from vllm import LLM, SamplingParams
import pandas as pd
import wandb
from transformers import AutoTokenizer

# system_prompt = """You will be given the responses of two models (identified by their huggingface model name) for the same prompt. Please list the differences between the two responses that a user might be interested in. This could pertrain to things like capability, thought process, style, tone, formatting, safety, etc. Please be as specific as possible."""
system_prompt_short = """You are an expert model behavior analyst. You will be given:
1. The original user prompt
2. A response from Model A (with model name)
3. A response from Model B (with model name)

Your task is to clearly and concisely list key differences between the two responses that would impact a user's experience. Focus on high-level properties such as capability, reasoning approach, tone, style, formatting, safety, or helpfulness. Do not quote or summarize specific content—only describe response characteristics.

Assume the user cannot see the responses and is relying on you to understand how the models differ. Highlight only 3–5 major differences; if the responses are short, 1–2 may suffice. If there are no meaningful distinctions, respond with "no significant differences found" or "responses are roughly equivalent."

Avoid introductions or conclusions. Keep your response under 6 sentences (formed as list) and use model names (instead of Model A/Model B) when making comparisons."""

system_prompt = """You are an expert model behavior analyst. You will be given:
1. The original user prompt
2. Response from Model A (with model name)
3. Response from Model B (with model name)

Analyze and list the key differences between these responses. Consider the following dimensions:

1. Content & Reasoning:
   - Knowledge depth and accuracy
   - Reasoning approach and thoroughness
   - How instructions were interpreted/followed

2. Communication Style:
   - Tone and formality level
   - Conciseness vs. verbosity
   - Structure and formatting choices

3. User Experience:
   - Helpfulness for the specific context
   - Creativity or originality
   - Personality and engagement style

Please be clear and concise, focusing only on the most significant differences that would impact a user's experience. Your response should be brief - typically a few sentences and no more than 2 short paragraphs. Highlight only 3-5 major differences between the responses. For short responses, there may only be 1-2 meaningful differences or none at all. If no differences would meaningfully impact the user experience, simply state "no significant differences found" or "responses are roughly equivalent"."""

new_system_prompt = """You are an expert model behavior analyst. Your task is to meticulously compare two model responses to a given user prompt and identify key qualitative differences. For each significant difference, you must determine if it's more likely a **general trait** of the model or a **context-specific** behavior triggered by the current prompt.

**Prioritize conciseness and clarity in all your descriptions and explanations.** Aim for the most impactful information in the fewest words.

You will be provided with:
1.  **User Prompt:** The original prompt given to both models.
2.  **Model A Name:** The identifier for Model A.
3.  **Model A Response:** The response from Model A.
4.  **Model B Name:** The identifier for Model B.
5.  **Model B Response:** The response from Model B.

**Your Goal:**
Produce a JSON list of objects. Each object will represent a single distinct difference. Focus on impactful differences a user would notice. 

**Definitions:**
*   **General Trait:** Reflects a model's typical behavior across diverse prompts.
    *   *Think:* Is this how Model A *usually* is vs. Model B?
*   **Context-Specific Difference:** Arises mainly due to *this specific* user prompt.
    *   *Think:* Is this difference a direct reaction to *this current prompt*?
*   **Impact:** How much does this difference impact the user's experience?
    *   *Think:* Is this difference a major factor in the user's experience?
*   **Unexpected Behavior:** Do one of the model responses contain highly unusual or concerning behavior? If true then a developer will analyze these responses manually, so use this flag sparingly.
    *   *Think:* Does this difference involve offensive language, gibberish, bias, factual hallucinations, or other strange or unwanted behavior?

**JSON Output Structure for each difference (BE BRIEF, if no notable differences exist, return empty list):**
```json
[
  {
    "difference": "Brief description (max 2 sentences)",
    "category": "1-4 word category",
    "a_evidence": "Direct quote or evidence from Model A",
    "b_evidence": "Direct quote or evidence from Model B",
    "type": "General|Context-Specific",
    "reason": "Brief justification (max 2 sentences)",
    "impact": "Low|Medium|High",
    "unexpected_behavior": "True|False"
  }
]
```

**Example JSON Output:**
```json
[
    {
    "difference": "Model A used a formal tone, Model B was casual.",
    "category": "Tone",
    "a_evidence": "Quote: 'It is imperative to consider...'",
    "b_evidence": "Quote: 'Hey there! So, basically...'",
    "type": "General",
    "reason": "A consistently defaults to formal, B to informal, across various prompts.",
    "impact": "Medium",
    "unexpected_behavior": "False"
    },
    {
    "difference": "Model A used a functional programming approach, Model B used an object-oriented approach.",
    "category": "Coding Style",
    "a_evidence": "Uses map() and filter() to process the data.",
    "b_evidence": "Creates a class to encapsulate the data and methods.",
    "type": "Context-Specific",
    "reason": "The prompt asked for a solution to a specific data processing task; Model A's functional approach and Model B's object-oriented approach seem chosen based on the task's requirements and the models' understanding of the context.",
    "impact": "High",
    "unexpected_behavior": "False"
    },
    {
    "difference": "Model A provided specific citations and caveats, Model B stated facts without verification.",
    "category": "Fact Verification",
    "a_evidence": "Quote: 'According to the 2023 WHO report... However, this data may vary by region and should be cross-referenced.'",
    "b_evidence": "Quote: 'The global vaccination rate is 78% and continues to increase rapidly worldwide.'",
    "type": "General",
    "reason": "Model A consistently includes source attribution and uncertainty markers, while Model B presents information with absolute confidence.",
    "impact": "High",
    "unexpected_behavior": "False"
    }
]
```"""

one_sided_system_prompt = """You are an expert model behavior analyst. Your task is to meticulously compare two model responses to a given user prompt and identify unique qualitative properties belonging to one model but not the other. For each significant property, you must determine if it's more likely a **general trait** of the model or a **context-specific** behavior triggered by the current prompt.

**Prioritize conciseness and clarity in all your descriptions and explanations.** Aim for the most impactful information in the fewest words.

You will be provided with:
1.  **User Prompt:** The original prompt given to both models.
2.  **Model A Name:** The identifier for Model A.
3.  **Model A Response:** The response from Model A.
4.  **Model B Name:** The identifier for Model B.
5.  **Model B Response:** The response from Model B.

**Your Goal:**
Produce a JSON list of objects. Each object will represent a single distinct property observed in one model's response that is notably absent or different in the other's. Focus on identifying the impactful differences a user would notice. Do NOT include properties that related to detail (e.g. "Model A does a more detailed analysis than Model B") since this can already be measured by the number of tokens in the response.

**Definitions:**
*   **General Trait:** Reflects a model's typical behavior across diverse prompts.
    *   *Think:* Is this how this Model *usually* is compared to the other?
*   **Context-Specific Difference:** Arises mainly due to *this specific* user prompt.
    *   *Think:* Is this property a direct reaction to *this current prompt*?
*   **Impact:** How much does this property impact the user's experience?
    *   *Think:* Is this property a major factor in the user's experience?
*   **Unexpected Behavior:** Does the model's response contain highly unusual or concerning behavior? If true then a developer will analyze these responses manually, so use this flag sparingly.
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
    "unexpected_behavior": "True|False"
  }
]
```

**Example JSON Output:**
```json
[
  {
    "model": "{{Model A Name}}",
    "property_description": "formal and professional tone.",
    "category": "Tone",
    "evidence": "Quote: 'It is imperative to consider the implications...'",
    "type": "General",
    "reason": "{{Model A Name}}'s response is in a formal register, which is a notable contrast to {{Model B Name}}'s more casual style.",
    "impact": "Medium",
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model B Name}}",
    "property_description": "casual and conversational tone.",
    "category": "Tone",
    "evidence": "Quote: 'Hey there! So, basically, what you gotta think about is...'",
    "type": "General",
    "reason": "{{Model B Name}}'s response is in an informal, friendly style, which stands out compared to {{Model A Name}}'s formality.",
    "impact": "Medium",
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
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model A Name}}",
    "property_description": "cautious approach to factual claims.",
    "category": "Fact Verification",
    "evidence": "Quote: 'According to the 2023 WHO report... However, this data may vary by region and should be cross-referenced.'",
    "type": "General",
    "reason": "{{Model A Name}} prioritizes accuracy and uncertainty, providing source attribution and disclaimers, unlike {{Model B Name}}'s direct factual statements.",
    "impact": "High",
    "unexpected_behavior": "False"
  },
  {
    "model": "{{Model B Name}}",
    "property_description": "factual information with high confidence and without explicit verification or caveats.",
    "category": "Fact Verification",
    "evidence": "Quote: 'The global vaccination rate is 78% and continues to increase rapidly worldwide.'",
    "type": "General",
    "reason": "{{Model B Name}} states facts without providing sources or acknowledging potential variability, contrasting with {{Model A Name}}'s cautious approach.",
    "impact": "High",
    "unexpected_behavior": "False"
  }
]
```"""

def format_example(df, prompt):
    df_prompt = df[df["prompt"] == prompt]
    assert len(df_prompt) == 2, "Expected 2 responses for prompt"
    return f"# Prompt: {prompt}\n\n# {df_prompt['model'].iloc[0]} response: {df_prompt['model_response'].iloc[0]}\n\n--------------------------------\n\n# {df_prompt['model'].iloc[1]} response: {df_prompt['model_response'].iloc[1]}"

def remove_thinking_from_output(output):
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare model responses and identify differences')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of prompts to process (default: process all prompts)')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-32B",
                       help='Model name to use for the LLM')
    parser.add_argument('--output_file', type=str, default="disguising/misc/model_comparison_differences_shorter.csv",
                       help='Output file to save the results')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top p')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='Max tokens')
    parser.add_argument('--max_model_len', type=int, default=16384,
                       help='Max model length')
    parser.add_argument('--model1_file', type=str, default="disguising/model-responses/base_500_all_models/meta-llama_Llama-3.1-8B-Instruct.csv",
                       help='Model 1 file')
    parser.add_argument('--model2_file', type=str, default="disguising/model-responses/base_500_all_models/microsoft_phi-4_responses.csv",
                       help='Model 2 file')
    parser.add_argument('--model1_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Model 1 name')
    parser.add_argument('--model2_name', type=str, default="microsoft/phi-4",
                       help='Model 2 name')
    args = parser.parse_args()

    # Use only the JSON system prompt
    selected_system_prompt = one_sided_system_prompt

    run = wandb.init(project="difference-training", name="data_gen")
    run.summary["system_prompt"] = selected_system_prompt
    run.config["model_name"] = args.model_name
    run.config["output_file"] = args.output_file
    run.config["num_samples"] = args.num_samples
    run.config["system_prompt_type"] = "one_sided_json"

    # make output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    df1 = pd.read_csv(args.model1_file).drop_duplicates(subset=["prompt", "model_response"])
    df1["prompt"] = df1["prompt"].astype(str).str.replace(r'^"|"$', '', regex=True).str.strip()
    df1["model"] = args.model1_name
    df2 = pd.read_csv(args.model2_file).drop_duplicates(subset=["prompt", "model_response"])
    df2["prompt"] = df2["prompt"].astype(str).str.replace(r'^"|"$', '', regex=True).str.strip()
    df2["model"] = args.model2_name
    df = pd.concat([df1, df2])
    prompts = df["prompt"].unique().tolist()
    models = df["model"].unique().tolist()
    print(f"Found {len(prompts)} prompts and {len(models)} models")

    # drop any propmt which do not have both models
    prompts = [prompt for prompt in prompts if len(df[df["prompt"] == prompt]) == 2]
    print(f"Found {len(prompts)} prompts after dropping prompts with less than 2 models")

    # Limit prompts if num_samples is specified
    if args.num_samples is not None:
        prompts = prompts[:args.num_samples]
        print(f"Processing only {len(prompts)} prompts (limited by num_samples={args.num_samples})")

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

    # Process prompts in batches of 10 for periodic saving
    new_data = {"prompt": [], "model_1_response": [], "model_2_response": [], "model_1_name": [], "model_2_name": [], "differences": [], "parsed_differences": [], "parse_error": []}

    batch_size = 10
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_messages = []
        batch_data = {"prompt": [], "model_1_response": [], "model_2_response": [], "model_1_name": [], "model_2_name": []}
        
        print(f"Processing batch {batch_start//batch_size + 1}: prompts {batch_start+1}-{batch_end}")
        
        for prompt in batch_prompts:
            try:
                df_prompt = df[df["prompt"] == prompt]
                model_1_name = df_prompt['model'].iloc[0]
                model_2_name = df_prompt['model'].iloc[1]
                model_1_response = remove_thinking_from_output(df_prompt['model_response'].iloc[0])
                model_2_response = remove_thinking_from_output(df_prompt['model_response'].iloc[1])
                
                message = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": selected_system_prompt}, 
                        {"role": "user", "content": format_example(df, prompt)}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                batch_messages.append(message)
                
                # Store the data for this prompt
                batch_data["prompt"].append(prompt)
                batch_data["model_1_response"].append(model_1_response)
                batch_data["model_2_response"].append(model_2_response)
                batch_data["model_1_name"].append(model_1_name)
                batch_data["model_2_name"].append(model_2_name)
                
            except Exception as e:
                print(f"Error processing prompt: {str(prompt)[:100]}...")
                print(f"Error details: {str(e)}")
                print(f"Skipping this prompt and continuing...")
                continue
        
        # Get LLM responses for the batch
        batch_responses = llm.generate(batch_messages, sampling_params=sampling_params)
        
        # Process the responses
        for i, response in enumerate(batch_responses):
            cleaned_response = remove_thinking_from_output(response.outputs[0].text)
            parsed_json, parse_error = parse_json_response(cleaned_response)
            
            # Add to main data structure
            new_data["prompt"].append(batch_data["prompt"][i])
            new_data["model_1_response"].append(batch_data["model_1_response"][i])
            new_data["model_2_response"].append(batch_data["model_2_response"][i])
            new_data["model_1_name"].append(batch_data["model_1_name"][i])
            new_data["model_2_name"].append(batch_data["model_2_name"][i])
            new_data["differences"].append(cleaned_response)
            new_data["parsed_differences"].append(parsed_json)
            new_data["parse_error"].append(parse_error)
        
        # Create current results dataframe and save after each batch
        current_results_df = pd.DataFrame(new_data).dropna(subset=['prompt'])
        save_intermediate_results(current_results_df, args.output_file, batch_start//batch_size + 1)
        
        # Log successful parsing rate
        successful_parses = sum(1 for pe in new_data["parse_error"] if pe is None)
        total_processed = len(new_data["parse_error"])
        parsing_rate = successful_parses/total_processed*100 if total_processed > 0 else 0
        print(f"JSON parsing success rate so far: {successful_parses}/{total_processed} ({parsing_rate:.1f}%)")
        
        # Log parsing stats to wandb
        wandb.log({
            "parsing_success_rate": parsing_rate,
            "successful_parses": successful_parses,
            "total_processed": total_processed
        })

    # Create the final dataframe
    results_df = pd.DataFrame(new_data).dropna(subset=['prompt'])

    # Print sample results
    print("\nSample raw response:")
    print(results_df["differences"].iloc[0])
    print("\nSample parsed JSON:")
    print(results_df["parsed_differences"].iloc[0])
    print("--------------------------------")

    # Print parsing statistics
    successful_parses = results_df["parse_error"].isna().sum()
    total_responses = len(results_df)
    final_parsing_rate = successful_parses/total_responses*100 if total_responses > 0 else 0
    print(f"Final JSON parsing success rate: {successful_parses}/{total_responses} ({final_parsing_rate:.1f}%)")

    # Save the final dataframe (this will be the same as the last intermediate save)
    results_df.to_json(args.output_file.replace(".csv", ".jsonl"), orient='records', lines=True)
    print(f"Final results saved to {args.output_file}")
    print(f"Dataframe shape: {results_df.shape}")
    print(f"Columns: {list(results_df.columns)}")

    # cast all columns to string
    results_df = results_df.astype(str)

    #  Final wandb log with complete results
    run.log({"final_results": wandb.Table(dataframe=results_df.astype(str))})

    # Log final summary to wandb
    wandb.log({
        "final_parsing_success_rate": final_parsing_rate,
        "final_total_processed": total_responses,
        "final_successful_parses": successful_parses
    })

if __name__ == "__main__":
    main()

