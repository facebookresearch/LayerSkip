from typing import List, Optional, Dict
import torch
import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import forward_depth_adaptive_sequence, decode_next_token

class DepthAdaptiveSequenceGenerationStrategy(GenerationStrategy):
    """Sequence-level depth adaptive generation strategy.
    
    This strategy uses a sequence-level early exit mechanism where the entire
    sequence is processed up to a certain layer depth, determined dynamically
    based on prediction confidence.
    
    This implementation more closely aligns with the Depth-Adaptive Transformer paper:
    1. Uses geometric accumulation for halting probabilities
    2. Implements layer-specific thresholds for finer control
    3. Adds random seed for reproducibility
    """
    
    def __init__(
        self,
        halting_threshold: float = 0.9,
        min_layers: int = 4,
        max_layers: Optional[int] = None,
        layer_thresholds: Optional[Dict[int, float]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the sequence-level depth adaptive strategy.
        
        Args:
            halting_threshold: Base probability threshold to determine early exit
            min_layers: Minimum number of layers to process before considering early exit
            max_layers: Maximum number of layers to process (None = all layers)
            layer_thresholds: Dictionary mapping layer indices to specific thresholds
            seed: Random seed for reproducibility
        """
        self.halting_threshold = halting_threshold
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.layer_thresholds = layer_thresholds or {
            4: 0.85,  # Lower threshold at earlier eligible layers
            6: 0.85,
            8: 0.85,
        }
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
        """Generate tokens using sequence-level depth adaptation."""
        past_key_values = None
        input_ids = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []
        
        # Use parameters from generation config if provided
        halting_threshold = generation_config.halting_threshold if hasattr(generation_config, 'halting_threshold') else self.halting_threshold
        min_layers = generation_config.min_layers if hasattr(generation_config, 'min_layers') else self.min_layers
        max_layers = generation_config.max_layers if hasattr(generation_config, 'max_layers') else self.max_layers
        # Use same seed for all generation steps or increment for each step
        seed = generation_config.seed if hasattr(generation_config, 'seed') else self.seed
        
        for step in range(generation_config.max_steps):
            # Forward pass with depth adaptation
            step_seed = seed + step if seed is not None else None
            model_output = forward_depth_adaptive_sequence(
                model,
                input_ids,
                past_key_values,
                halting_threshold=halting_threshold,
                min_layers=min_layers,
                max_layers=max_layers,
                seed=step_seed,
            )
            
            # Extract logits and process if needed
            logits = model_output.logits
            
            if logits_processors:
                logits = logits_processors(input_ids, logits)
                
            past_key_values = model_output.past_key_values
            
            # Decode next token
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
            input_ids = torch.tensor([[next_token]]).to(model.device)
            
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,  # Not applicable for this strategy
        ) 