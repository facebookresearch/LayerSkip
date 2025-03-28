from typing import List, Optional
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
        # Convert input_ids to tensor and move to device.
        input_ids_tensor = torch.tensor([input_ids]).to(model.device)  # Shape: [1, seq_len]
        output_ids: List[int] = []

        for step in range(generation_config.max_steps):
            # Run token-level forward adaptive function.
            final_hidden_states, exit_mask = forward_depth_adaptive_token(
                model,
                input_ids_tensor,
                halting_threshold=self.halting_threshold,
                min_layers=self.min_layers,
                max_layers=self.max_layers,
            )
            # Obtain logits from final hidden states.
            logits = model.lm_head(final_hidden_states)
            if logits_processors:
                logits = logits_processors(input_ids_tensor, logits)
            
            next_token, _ = decode_next_token(
                logits=logits,
                token_idx=-1,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if next_token in eos_token_ids:
                break
            if stopping_criteria and torch.all(stopping_criteria(input_ids_tensor, scores=None)):
                break
            output_ids.append(next_token)
            # Prepare input for the next generation step.
            input_ids_tensor = torch.tensor([[next_token]]).to(model.device)
        
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )
