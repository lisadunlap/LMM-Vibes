import argparse
import wandb
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainerCallback
import json
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train difference generator model')
parser.add_argument('--learning_rate', type=float, default=5e-5, 
                    help='Learning rate for training (default: 5e-5)')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B",
                    help='Model name for training (default: Qwen/Qwen3-8B)')
parser.add_argument('--data_file', type=str, default="disguising/misc/model_comparison_differences_14b-short-llama-phi-list.csv",
                    help='Data file for training (default: disguising/misc/model_comparison_differences_14b-short-llama-phi-list.csv)')
parser.add_argument('--train_on_responses', type=bool, default=False,
                    help='Whether to train on responses only (default: True)')
args = parser.parse_args()

print(f"Using learning rate: {args.learning_rate}")

systems_prompt = """You are an expert model behavior analyst. Given a prompt and the names of two models, your task is to predict how the models would likely differ in their responses. Focus on high-level properties such as capability, reasoning approach, tone, style, formatting, safety, or helpfulness. Do not quote or summarize specific content—only describe response characteristics.

Highlight only 3–5 major differences the models would likely exhibit; if the prompt is simple, 1–2 may suffice. If you expect no meaningful distinctions, respond with "no significant differences expected" or "responses would likely be roughly equivalent."

Avoid introductions or conclusions. Keep your response under 6 sentences (formed as list) and use the specific model names when making comparisons."""

data_file = "disguising/misc/model_comparison_differences_14b-short-llama-phi-list.csv"
differences = pd.read_csv(data_file)
print(differences.head())

def generate_conversation(examples):
    prompts = examples["prompt"]
    differences = examples["differences"]
    conversations = []
    for prompt, difference in zip(prompts, differences):
        conversations.append([
            {"role": "system", "content": systems_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": difference},
        ])
    return {"conversations": conversations}

# Create dataset from pandas dataframe
dataset = Dataset.from_pandas(differences)

# Apply conversation formatting
dataset = dataset.map(generate_conversation, batched=True)

# Split dataset into train (80%) and test (20%)
dataset_split = dataset.train_test_split(test_size=0.2, seed=3407)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

run = wandb.init(project="difference-generator", name="difference-generator")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = 2048,   # Context length - can be longer, but uses more memory
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = True, # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)

def formatting_prompts_func(examples):
    """Format prompts for training."""
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Apply text formatting after tokenizer is available
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
print(train_dataset[0])

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

from litellm import completion
def eval_responses(response, ground_truth):
    """Evaluate the response against the ground truth."""
    prompt = f"""You are a fair judge that will compare how similar a response is to the ground truth. The task that the responses are for is to predict how two models would differ in their responses to a given prompt. The response is the prediction of the model, and the ground truth is the actual difference between the two models. 
    
    Please return 2 scores:
    - similarity score (1-5, where 5 is very similar): how similar the response is to the ground truth. Think about if the response and ground truth are mentioning about the same differences. Are there any points where the prediction contradicts the ground truth? Are there any points where the prediction is missing a difference that the ground truth mentions? Is there any point where the prediction is mentioning a difference that the ground truth does not? Only give a score of 1 if none of the differences mentioned in the ground truth are mentioned in the response.
    - hallucination score (1-5, where 5 is no hallucinations): whether the response is hallucinating. Are there any points where the prediction is stating specific attributes that the models have that are not mentioned in the ground truth? For example, if the response mentions that one madel says Y in its response, but the ground truth does not mention that model saying Y, then this is a hallucination.
    
    The response is: {response}
    The ground truth is: {ground_truth}

    Please return your scores in the following format with no other text:
    {{"similarity_score": [SCORE], "hallucination_score": [SCORE]}}
    """

    response = completion(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    output = response["choices"][0]["message"]["content"]
    try:
        parsed_output = json.loads(output)
        return parsed_output["similarity_score"], parsed_output["hallucination_score"]
    except:
        print(f"Error parsing output: {output}")
        return 0, 0

class PredictionLoggingCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, model):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.eval_interval = 50  # Evaluate every 50 steps
        
        # Track best metrics
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_overall_accuracy = 0.0
        self.best_good_prediction_rate = 0.0
        self.best_similarity_score = 0.0
        self.best_hallucination_score = 0.0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Track training loss if available in logs
        if logs and 'train_loss' in logs:
            current_train_loss = logs['train_loss']
            if current_train_loss < self.best_train_loss:
                self.best_train_loss = current_train_loss
                # Log the best training loss
                wandb.log({"best_train_loss": self.best_train_loss}, step=state.global_step)
        
        # Run evaluation every eval_interval steps
        if state.global_step % self.eval_interval == 0 and state.global_step > 0:
            self.run_evaluation(state)
    
    def run_evaluation(self, state):
        # Calculate evaluation loss manually
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Calculate eval loss on a subset of eval data
        # eval_sample_size = min(20, len(self.eval_dataset))
        eval_sample_size = len(self.eval_dataset)
        eval_indices = np.random.choice(len(self.eval_dataset), eval_sample_size, replace=False)
        
        with torch.no_grad():
            for idx in eval_indices:
                text = self.eval_dataset[int(idx)]['text']
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
                
                # Calculate loss
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_eval_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Update best validation loss
        if avg_eval_loss < self.best_val_loss:
            self.best_val_loss = avg_eval_loss
        
        # Log validation loss and best validation loss
        wandb.log({
            "val_loss": avg_eval_loss,
            "best_val_loss": self.best_val_loss
        }, step=state.global_step)
        
        # Generate predictions on a subset of eval data for logging
        sample_size = min(100, len(self.eval_dataset))
        sample_indices = np.random.choice(len(self.eval_dataset), sample_size, replace=False)
        
        predictions_data = []
        similarity_scores = []
        hallucination_scores = []
        
        for idx in sample_indices:
            # Get the conversation from eval dataset
            conversation = self.eval_dataset[int(idx)]['conversations']
            
            # Extract prompt and ground truth
            prompt = conversation[1]['content']  # User message
            ground_truth = conversation[2]['content']  # Assistant response
            
            # Create input for generation (system + user message)
            messages = [
                {"role": "system", "content": systems_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Generate prediction
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode prediction (remove input tokens)
            prediction = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )

            similarity_score, hallucination_score = eval_responses(prediction, ground_truth)
            
            # Collect scores for aggregate metrics
            similarity_scores.append(similarity_score)
            hallucination_scores.append(hallucination_score)

            predictions_data.append({
                "prompt": prompt,
                "prediction": prediction.strip(),
                "ground_truth": ground_truth,
                "step": state.global_step,
                "similarity_score": similarity_score,
                "hallucination_score": hallucination_score
            })
        
        # Calculate and log aggregate accuracy metrics
        if similarity_scores and hallucination_scores:
            avg_similarity = np.mean(similarity_scores)
            avg_hallucination = np.mean(hallucination_scores)
            
            # Calculate overall accuracy (weighted combination of similarity and hallucination scores)
            # Higher similarity score is better, higher hallucination score is also better (less hallucination)
            # Normalize to 0-1 scale and take weighted average
            similarity_normalized = (avg_similarity - 1) / 4  # Convert 1-5 scale to 0-1
            hallucination_normalized = (avg_hallucination - 1) / 4  # Convert 1-5 scale to 0-1
            overall_accuracy = (similarity_normalized * 0.7 + hallucination_normalized * 0.3)  # Weight similarity more
            
            # Calculate percentage of "good" predictions (similarity >= 4 and hallucination >= 4)
            good_predictions = sum(1 for s, h in zip(similarity_scores, hallucination_scores) if s >= 4 and h >= 4)
            good_prediction_rate = good_predictions / len(similarity_scores)
            
            # Update best metrics
            if overall_accuracy > self.best_overall_accuracy:
                self.best_overall_accuracy = overall_accuracy
            if good_prediction_rate > self.best_good_prediction_rate:
                self.best_good_prediction_rate = good_prediction_rate
            if avg_similarity > self.best_similarity_score:
                self.best_similarity_score = avg_similarity
            if avg_hallucination > self.best_hallucination_score:
                self.best_hallucination_score = avg_hallucination
            
            # Log current and best metrics
            wandb.log({
                "eval_avg_similarity_score": avg_similarity,
                "eval_avg_hallucination_score": avg_hallucination,
                "eval_overall_accuracy": overall_accuracy,
                "eval_good_prediction_rate": good_prediction_rate,
                "eval_num_samples": len(similarity_scores),
                # Best metrics
                "best_overall_accuracy": self.best_overall_accuracy,
                "best_good_prediction_rate": self.best_good_prediction_rate,
                "best_similarity_score": self.best_similarity_score,
                "best_hallucination_score": self.best_hallucination_score
            }, step=state.global_step)
            
            print(f"Step {state.global_step} - Avg Similarity: {avg_similarity:.2f}, Avg Hallucination: {avg_hallucination:.2f}, Overall Accuracy: {overall_accuracy:.3f}, Good Prediction Rate: {good_prediction_rate:.3f}")
            print(f"Best Metrics - Val Loss: {self.best_val_loss:.4f}, Train Loss: {self.best_train_loss:.4f}, Overall Accuracy: {self.best_overall_accuracy:.3f}, Good Prediction Rate: {self.best_good_prediction_rate:.3f}")
        
        # Log predictions table to wandb
        if predictions_data:
            predictions_table = wandb.Table(
                columns=["prompt", "prediction", "ground_truth", "step", "similarity_score", "hallucination_score"],
                data=[[row["prompt"], row["prediction"], row["ground_truth"], row["step"], row["similarity_score"], row["hallucination_score"]] 
                      for row in predictions_data]
            )
            wandb.log({"predictions_table": predictions_table}, step=state.global_step)
        
        # Set model back to training mode
        self.model.train()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final summary metrics at the end of training."""
        summary_metrics = {
            "final_best_train_loss": self.best_train_loss,
            "final_best_val_loss": self.best_val_loss,
            "final_best_overall_accuracy": self.best_overall_accuracy,
            "final_best_good_prediction_rate": self.best_good_prediction_rate,
            "final_best_similarity_score": self.best_similarity_score,
            "final_best_hallucination_score": self.best_hallucination_score
        }
        
        # Log as summary
        for key, value in summary_metrics.items():
            wandb.run.summary[key] = value
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY - BEST METRICS:")
        print("="*50)
        print(f"Best Training Loss: {self.best_train_loss:.4f}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Overall Accuracy: {self.best_overall_accuracy:.3f}")
        print(f"Best Good Prediction Rate: {self.best_good_prediction_rate:.3f}")
        print(f"Best Similarity Score: {self.best_similarity_score:.2f}")
        print(f"Best Hallucination Score: {self.best_hallucination_score:.2f}")
        print("="*50)

# Initialize callback
prediction_callback = PredictionLoggingCallback(eval_dataset, tokenizer, model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,  # Now using the eval dataset
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,  # Add eval batch size
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = args.learning_rate, # Use argparse learning rate
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "wandb", # Use this for WandB etc
        max_grad_norm = 1.0,
    ),
    callbacks=[prediction_callback],  # Add our custom callback
)

if args.train_on_responses:
    trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )

trainer.train()

# Test the final model
messages = [
    {"role" : "system", "content" : systems_prompt},
    {"role" : "user", "content" : "Who are you?"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 512, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)