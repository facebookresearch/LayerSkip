from typing import List, Optional
import torch
import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import forward_with_layerdrop, decode_next_token

class LayerDropGenerationStrategy(GenerationStrategy):
    def __init__(self, dropout_rate: float = 0.2, seed: Optional[int] = None):
        self.dropout_rate = dropout_rate
        self.seed = seed
        
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
        """Generate tokens using LayerDrop."""
        past_key_values = None
        input_ids = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []
        #print(f"Input IDs: {input_ids.shape}, Past Key Values: {past_key_values.shape}")
        # Use current dropout rate from generation config if provided
        dropout_rate = generation_config.dropout_rate if hasattr(generation_config, 'dropout_rate') else self.dropout_rate

        for step in range(generation_config.max_steps):
            # Use LayerDrop forward function
            model_output = forward_with_layerdrop(
                model,
                input_ids,
                past_key_values,
                dropout_rate=dropout_rate,
                time_step=step,  # Current step for curriculum
                max_time_steps=generation_config.max_steps,  # Total steps
                seed=self.seed,
            )
            
            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
                
            past_key_values = model_output.past_key_values
            
            next_token, _ = decode_next_token(
                logits=logits,
                token_idx=-1,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p
            )
            
            if streamer:
                streamer.put(next_token)
                
            next_token = next_token.item()
            if next_token in eos_token_ids:
                break
                
            if stopping_criteria and torch.all(stopping_criteria(input_ids, scores=None)):
                break
                
            output_ids.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(input_ids)
            print("Checking output ids:",output_ids)
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,  # Not applicable for LayerDrop
        ) 