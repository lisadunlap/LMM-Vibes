"""
General utilities for parallel LLM and embedding calls with caching support.

This module provides reusable functions that eliminate code duplication across 
the codebase for parallel LLM operations while maintaining order preservation
and robust error handling.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import litellm
import numpy as np
from tqdm import tqdm
import logging

from .caching import LMDBCache

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    model: str = "gpt-4o-mini"
    max_workers: int = 10
    max_retries: int = 3
    base_sleep_time: float = 2.0
    timeout: Optional[float] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    

@dataclass
class EmbeddingConfig:
    """Configuration for embedding calls."""
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_workers: int = 10
    max_retries: int = 3
    base_sleep_time: float = 2.0


class LLMUtils:
    """Utility class for parallel LLM operations with caching."""
    
    def __init__(self, cache: Optional[LMDBCache] = None):
        """Initialize with optional cache instance."""
        self.cache = cache
        self._lock = threading.Lock()
    
    def parallel_completions(
        self,
        messages: List[Union[str, List[Dict[str, Any]]]],
        config: LLMConfig,
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
        progress_desc: str = "LLM calls"
    ) -> List[str]:
        """
        Execute LLM completions in parallel with order preservation.
        
        Args:
            messages: List of user messages (strings) or full message lists
            config: LLM configuration
            system_prompt: Optional system prompt (if messages are strings)
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar
            
        Returns:
            List of completion responses in the same order as input
        """
        if not messages:
            return []
            
        # Pre-allocate results to preserve order
        results: List[str] = [""] * len(messages)
        
        def _single_completion(idx: int, message: Union[str, List[Dict[str, Any]]]) -> Tuple[int, str]:
            """Process a single completion with retries."""
            for attempt in range(config.max_retries):
                try:
                    # Build request data
                    if isinstance(message, str):
                        request_messages = []
                        if system_prompt:
                            request_messages.append({"role": "system", "content": system_prompt})
                        request_messages.append({"role": "user", "content": message})
                    else:
                        request_messages = message
                    
                    request_data = {
                        "model": config.model,
                        "messages": request_messages,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                    }
                    if config.max_tokens:
                        request_data["max_completion_tokens"] = config.max_tokens
                    
                    # Check cache first
                    if self.cache:
                        cached_response = self.cache.get_completion(request_data)
                        if cached_response is not None:
                            return idx, cached_response["choices"][0]["message"]["content"]
                    
                    # Make API call
                    response = litellm.completion(
                        **request_data,
                        caching=False,  # Use our own caching
                        timeout=config.timeout
                    )
                    
                    content = response.choices[0].message.content
                    
                    # Cache the response
                    if self.cache:
                        response_dict = {
                            "choices": [{
                                "message": {"content": content}
                            }]
                        }
                        self.cache.set_completion(request_data, response_dict)
                    
                    return idx, content
                    
                except Exception as e:
                    if attempt == config.max_retries - 1:
                        logger.error(f"LLM call failed after {config.max_retries} attempts: {e}")
                        return idx, f"ERROR: {e}"
                    else:
                        sleep_time = config.base_sleep_time * (2 ** attempt)
                        logger.warning(f"LLM call attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
                        time.sleep(sleep_time)
            
            return idx, "ERROR: Max retries exceeded"
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(_single_completion, idx, msg): idx 
                for idx, msg in enumerate(messages)
            }
            
            # Process results with optional progress bar
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(messages), desc=progress_desc)
                
            for future in iterator:
                idx, result = future.result()
                results[idx] = result
        
        return results
    
    def parallel_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig,
        show_progress: bool = True,
        progress_desc: str = "Embedding calls"
    ) -> List[List[float]]:
        """
        Generate embeddings in parallel with batching and order preservation.
        
        Args:
            texts: List of texts to embed
            config: Embedding configuration
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar
            
        Returns:
            List of embeddings in the same order as input texts
        """
        if not texts:
            return []
        
        # Pre-allocate results to preserve order
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        
        # Create batches with their positions
        batches = []
        for start in range(0, len(texts), config.batch_size):
            end = min(start + config.batch_size, len(texts))
            batch_texts = texts[start:end]
            batches.append((start, batch_texts))
        
        def _single_embedding_batch(start_idx: int, batch_texts: List[str]) -> Tuple[int, List[List[float]]]:
            """Process a single batch of embeddings with retries."""
            for attempt in range(config.max_retries):
                try:
                    # Check cache for each text in batch
                    batch_embeddings = []
                    uncached_indices = []
                    uncached_texts = []
                    
                    for i, text in enumerate(batch_texts):
                        if self.cache:
                            cached_embedding = self.cache.get_embedding(text)
                            if cached_embedding is not None:
                                batch_embeddings.append(cached_embedding.tolist())
                                continue
                        
                        # Mark as needing API call
                        batch_embeddings.append(None)
                        uncached_indices.append(i)
                        uncached_texts.append(text)
                    
                    # Make API call for uncached texts
                    if uncached_texts:
                        response = litellm.embedding(
                            model=config.model,
                            input=uncached_texts
                        )
                        
                        # Fill in uncached results
                        for i, embedding_data in enumerate(response.data):
                            batch_idx = uncached_indices[i]
                            embedding = embedding_data.embedding
                            batch_embeddings[batch_idx] = embedding
                            
                            # Cache the embedding
                            if self.cache:
                                self.cache.set_embedding(uncached_texts[i], np.array(embedding))
                    
                    return start_idx, batch_embeddings
                    
                except Exception as e:
                    if attempt == config.max_retries - 1:
                        logger.error(f"Embedding batch failed after {config.max_retries} attempts: {e}")
                        # Return zero embeddings as fallback
                        fallback_embeddings = [[0.0] * 1536] * len(batch_texts)  # Default to 1536 dims
                        return start_idx, fallback_embeddings
                    else:
                        sleep_time = config.base_sleep_time * (2 ** attempt)
                        logger.warning(f"Embedding batch attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
                        time.sleep(sleep_time)
            
            # Fallback
            fallback_embeddings = [[0.0] * 1536] * len(batch_texts)
            return start_idx, fallback_embeddings
        
        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(_single_embedding_batch, start, batch_texts): (start, len(batch_texts))
                for start, batch_texts in batches
            }
            
            # Process results with optional progress bar
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(batches), desc=progress_desc)
            
            for future in iterator:
                start_idx, batch_embeddings = future.result()
                batch_size = len(batch_embeddings)
                embeddings[start_idx:start_idx + batch_size] = batch_embeddings
        
        # Verify all embeddings were filled
        if any(e is None for e in embeddings):
            raise RuntimeError("Some embeddings are missing - check logs for errors.")
        
        return embeddings
    
    def single_completion(
        self,
        message: Union[str, List[Dict[str, Any]]],
        config: LLMConfig,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Single completion call with caching (convenience method).
        
        Args:
            message: User message (string) or full message list
            config: LLM configuration
            system_prompt: Optional system prompt (if message is string)
            
        Returns:
            Completion response
        """
        results = self.parallel_completions([message], config, system_prompt, show_progress=False)
        return results[0]


# Global instance with default cache
_default_cache = None
_default_llm_utils = None

def get_default_llm_utils() -> LLMUtils:
    """Get default LLMUtils instance with shared cache."""
    global _default_cache, _default_llm_utils
    
    if _default_llm_utils is None:
        if _default_cache is None:
            _default_cache = LMDBCache()
        _default_llm_utils = LLMUtils(_default_cache)
    
    return _default_llm_utils


# Convenience functions for common use cases
def parallel_completions(
    messages: List[Union[str, List[Dict[str, Any]]]],
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
    max_workers: int = 10,
    show_progress: bool = True,
    progress_desc: str = "LLM calls",
    **kwargs
) -> List[str]:
    """Convenience function for parallel completions with default settings."""
    # Separate function-specific parameters from config parameters
    config_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['max_retries', 'base_sleep_time', 'timeout', 'temperature', 'top_p', 'max_tokens']}
    
    config = LLMConfig(model=model, max_workers=max_workers, **config_kwargs)
    utils = get_default_llm_utils()
    return utils.parallel_completions(messages, config, system_prompt, show_progress, progress_desc)


def parallel_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_workers: int = 10,
    show_progress: bool = True,
    progress_desc: str = "Embedding calls",
    **kwargs
) -> List[List[float]]:
    """Convenience function for parallel embeddings with default settings."""
    # Separate function-specific parameters from config parameters
    config_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['max_retries', 'base_sleep_time']}
    
    config = EmbeddingConfig(model=model, batch_size=batch_size, max_workers=max_workers, **config_kwargs)
    utils = get_default_llm_utils()
    return utils.parallel_embeddings(texts, config, show_progress, progress_desc)


def single_completion(
    message: Union[str, List[Dict[str, Any]]],
    model: str = "gpt-4o-mini", 
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function for single completion with caching."""
    # Separate function-specific parameters from config parameters (no function-specific params for single)
    config_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['max_workers', 'max_retries', 'base_sleep_time', 'timeout', 'temperature', 'top_p', 'max_tokens']}
    
    config = LLMConfig(model=model, **config_kwargs)
    utils = get_default_llm_utils()
    return utils.single_completion(message, config, system_prompt)
