"""
OpenAI-based property extraction stage.

This stage migrates the logic from generate_differences.py into the pipeline architecture.
"""

from typing import Callable, Optional, List
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Property
from ..core.mixins import LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin
from ..prompts import extractor_prompts as _extractor_prompts
from ..core.caching import LMDBCache


class OpenAIExtractor(LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin, PipelineStage):
    """
    Extract behavioral properties using OpenAI models.
    
    This stage takes conversations and extracts structured properties describing
    model behaviors, differences, and characteristics.
    """
    
    def __init__(
        self,
        model: str = "gpt-4.1",
        system_prompt: str = "one_sided_system_prompt_no_examples",
        prompt_builder: Optional[Callable] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 16000,
        max_workers: int = 16,
        cache_dir: str = ".cache/lmmvibes",
        **kwargs
    ):
        """
        Initialize the OpenAI extractor.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4o-mini")
            system_prompt: System prompt for property extraction
            prompt_builder: Optional custom prompt builder function
            temperature: Temperature for LLM
            top_p: Top-p for LLM
            max_tokens: Max tokens for LLM
            max_workers: Max parallel workers for API calls
            cache_dir: Directory for LMDB cache
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.model = model
        # Allow caller to pass the name of a prompt template or the prompt text itself
        if isinstance(system_prompt, str) and hasattr(_extractor_prompts, system_prompt):
            self.system_prompt = getattr(_extractor_prompts, system_prompt)
        else:
            self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder or self._default_prompt_builder
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.cache = LMDBCache(cache_dir=cache_dir)

    def __del__(self):
        """Cleanup LMDB cache on deletion."""
        if hasattr(self, 'cache'):
            self.cache.close()

    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Run OpenAI extraction for all conversations.

        Each conversation is formatted with ``prompt_builder`` and sent to the
        OpenAI model in parallel using a thread pool.  The raw LLM response is
        stored inside a *placeholder* ``Property`` object (one per
        conversation).  Down-stream stages (``LLMJsonParser``) will parse these
        raw strings into fully-formed properties.
        """

        n_conv = len(data.conversations)
        if n_conv == 0:
            self.log("No conversations found – skipping extraction")
            return data

        self.log(f"Extracting properties from {n_conv} conversations using {self.model}")

        # ------------------------------------------------------------------
        # 1️⃣  Build user messages for every conversation
        # ------------------------------------------------------------------
        user_messages: List[str] = []
        for conv in data.conversations:
            user_messages.append(self.prompt_builder(conv))

        # ------------------------------------------------------------------
        # 2️⃣  Call the OpenAI API in parallel batches
        # ------------------------------------------------------------------
        raw_responses = self._extract_properties_batch(user_messages)

        # ------------------------------------------------------------------
        # 3️⃣  Wrap raw responses in placeholder Property objects
        # ------------------------------------------------------------------
        properties: List[Property] = []
        for conv, raw in zip(data.conversations, raw_responses):
            # We don't yet know which model(s) the individual properties will
            # belong to; parser will figure it out.  Use a placeholder model
            # name so that validation passes.
            prop = Property(
                id=str(uuid.uuid4()),
                question_id=conv.question_id,
                model=conv.model,   
                raw_response=raw,
            )
            properties.append(prop)

        self.log(f"Received {len(properties)} LLM responses")

        # Log to wandb if enabled
        if hasattr(self, 'use_wandb') and self.use_wandb:
            self._log_extraction_to_wandb(user_messages, raw_responses, data.conversations)

        # ------------------------------------------------------------------
        # 4️⃣  Return updated dataset
        # ------------------------------------------------------------------
        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=properties,
            clusters=data.clusters,
            model_stats=data.model_stats,
        )

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------

    def _call_openai_api(self, message: str) -> str:
        """Single threaded call with basic retry / error handling."""
        try:
            # Build request data for caching
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }

            # Check cache first
            cached_response = self.cache.get_completion(request_data)
            if cached_response is not None:
                self.log("Cache hit!", level="debug")
                return cached_response["choices"][0]["message"]["content"]

            # Call API if not in cache
            self.log("Cache miss - calling API", level="debug")
            response = litellm.completion(
                **request_data,
                caching=False,  # Disable litellm caching since we're using our own
            )
            
            # Cache the response
            response_dict = {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
            self.cache.set_completion(request_data, response_dict)
            
            return response.choices[0].message.content
        except litellm.ContextWindowExceededError as e:
            self.handle_error(e, "Context window exceeded – returning error string")
            return "ERROR: Context window exceeded. Input is too long."
        except Exception as e:
            # litellm may wrap OpenAI errors; include the message for debugging
            self.handle_error(e, "OpenAI API call failed – returning error string")
            return f"ERROR: {e}"

    def _extract_properties_batch(self, messages: List[str]) -> List[str]:
        """Call OpenAI for *messages* in parallel and preserve order."""

        results: List[str] = ["" for _ in messages]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._call_openai_api, msg): idx for idx, msg in enumerate(messages)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    # _call_openai_api already logged the error; store generic msg
                    results[idx] = f"ERROR: {e}"
        return results
    
    def _default_prompt_builder(self, conversation) -> str:
        """
        Default prompt builder for side-by-side comparisons.
        
        Args:
            conversation: ConversationRecord
            
        Returns:
            Formatted prompt string
        """
        if len(conversation.responses) == 2:
            # Side-by-side format
            model_a, model_b = conversation.model
            response_a = conversation.responses[0]
            response_b = conversation.responses[1]
            scores = conversation.scores

            # if scores is an empty dict, then we don't have scores
            if not scores:
                return (
                    f"# Prompt: {conversation.prompt}\n\n"
                    f"# {model_a} response:\n {response_a}\n\n"
                    f"# {model_b} response:\n {response_b}"
                )
            else:
                return (
                    f"# Prompt: {conversation.prompt}\n\n"
                    f"# {model_a} response:\n {response_a}\n\n"
                    f"# {model_b} response:\n {response_b}\n\n"
                    f"# Scores:\n {scores}"
                )
            
        else:
            # Single model format
            model = conversation.model
            response = conversation.responses
            scores = conversation.scores

            if not scores:
                print("No scores found")
                return (
                    f"# Prompt: {conversation.prompt}\n\n"
                    f"# Model response:\n {response}"
                )
            else:
                return (
                    f"# Prompt: {conversation.prompt}\n\n"
                    f"# Model response:\n {response}\n\n"
                    f"# Scores:\n {scores}"
                )
    
    def _log_extraction_to_wandb(self, user_messages: List[str], raw_responses: List[str], conversations):
        """Log extraction inputs/outputs to wandb."""
        try:
            import wandb
            
            # Create a table of inputs and outputs
            extraction_data = []
            for i, (msg, response, conv) in enumerate(zip(user_messages, raw_responses, conversations)):
                extraction_data.append({
                    "question_id": conv.question_id,
                    "input_message": msg,
                    "raw_response": response,
                    "response_length": len(response),
                    "has_error": response.startswith("ERROR:"),
                })
            
            # Log extraction table
            self.log_wandb({
                "extraction_inputs_outputs": wandb.Table(
                    columns=["question_id", "input_message", "raw_response", "response_length", "has_error"],
                    data=[[row[col] for col in ["question_id", "input_message", "raw_response", "response_length", "has_error"]] 
                          for row in extraction_data]
                )
            })
            
            # Log extraction metrics
            error_count = sum(1 for r in raw_responses if r.startswith("ERROR:"))
            self.log_wandb({
                "extraction_total_requests": len(raw_responses),
                "extraction_error_count": error_count,
                "extraction_success_rate": (len(raw_responses) - error_count) / len(raw_responses) if raw_responses else 0,
                "extraction_avg_response_length": sum(len(r) for r in raw_responses) / len(raw_responses) if raw_responses else 0,
            })
            
        except Exception as e:
            self.log(f"Failed to log extraction to wandb: {e}", level="warning")
    