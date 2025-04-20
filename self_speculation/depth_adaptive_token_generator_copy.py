from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import forward_depth_adaptive_token, decode_next_token

class DepthAdaptiveTokenGenerationStrategy(GenerationStrategy):
    def __init__(
        self,
        halting_threshold: float = 0.99,
        min_layers: int = 4,
        max_layers: Optional[int] = None,
    ):
        self.halting_threshold = halting_threshold
        self.min_layers = min_layers
        self.max_layers = max_layers

    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        past_key_values = None
        input_ids = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []
        
        for step in range(generation_config.max_steps):
            model_output = forward_depth_adaptive_token(
                model,
                input_ids,
                past_key_values,
                halting_threshold=self.halting_threshold,
                min_layers=self.min_layers,
                max_layers=self.max_layers,
            )
            
            logits = model_output.logits
            print(f"Step {step} - Raw logits shape:", logits.shape)
            
            # Take only the last token's logits while keeping 3D shape
            last_token_logits = logits[:, -1:, :]  # Shape: [batch_size, 1, vocab_size]
            print(f"Step {step} - Last token logits shape:", last_token_logits.shape)
            
            if logits_processors:
                last_token_logits = logits_processors(input_ids, last_token_logits)
            
            past_key_values = model_output.past_key_values
            
            # Apply temperature scaling before token selection
            if generation_config.temperature and generation_config.temperature != 1.0:
                scaled_logits = last_token_logits / generation_config.temperature
            else:
                scaled_logits = last_token_logits
            
            next_token, probs = decode_next_token(
                logits=scaled_logits,
                token_idx=-1,
                sample=generation_config.sample,
                temperature=1.0,  # We've already applied temperature
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )
            
            print(f"Step {step} - Selected token:", next_token.item())
            if probs is not None:
                top_probs, top_indices = probs.topk(5)
                print(f"Step {step} - Top 5 probabilities:", top_probs)
                print(f"Step {step} - Top 5 indices:", top_indices)
                print(f"Step {step} - Top 5 logits:", scaled_logits[0, 0, top_indices])  # Access 3D tensor correctly
            
            if streamer:
                streamer.put(next_token)
            
            next_token = next_token.item()
            if next_token in eos_token_ids:
                break
            
            if stopping_criteria and torch.all(stopping_criteria(input_ids, scores=None)):
                break
            
            output_ids.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(input_ids.device)
            print(f"Step {step} - New input_ids shape:", input_ids.shape)
        
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )
