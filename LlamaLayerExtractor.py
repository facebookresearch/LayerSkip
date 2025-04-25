# This module provides functions to extract intermediate layer representations
# from the LLaMA model for use with adapters

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
import transformers

class LlamaLayerExtractor:
    """
    A helper class that extracts hidden states from specified layers of a LLaMA model.
    This is used to feed intermediate representations to adapter modules.
    """
    
    def __init__(self, model: transformers.LlamaForCausalLM, target_layer: int = 8):
        """
        Initialize the layer extractor with a LLaMA model.
        
        Args:
            model: The LLaMA model to extract from
            target_layer: The layer index to extract (default: 8)
        """
        self.model = model
        self.target_layer = target_layer
        self.hidden_states = None
        
        # Register hooks to capture the output of the target layer
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on the target layer to capture its output."""
        target_module = self.model.model.layers[self.target_layer]
        
        def hook_fn(module, input, output):
            # Store the hidden states (output[0] contains the hidden states)
            self.hidden_states = output[0]
        
        # Register the hook on the final part of the layer
        # This will capture the complete output after all operations
        self.hook_handle = target_module.register_forward_hook(hook_fn)
    
    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
    
    def get_layer_output(self, input_ids: torch.Tensor, 
                         attention_mask: Optional[torch.Tensor] = None,
                         past_key_values: Optional[Tuple] = None) -> torch.Tensor:
        """
        Get the output of the target layer for the given input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            past_key_values: Optional past key values for faster generation
            
        Returns:
            Hidden states from the target layer
        """
        # Reset hidden states
        self.hidden_states = None
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,  # Ensure hidden states are computed
                return_dict=True
            )
        
        # Return the captured hidden states
        if self.hidden_states is None:
            raise ValueError(f"Failed to capture hidden states from layer {self.target_layer}")
            
        return self.hidden_states

# Helper function to extract layer 8 output directly from self-speculative decoding
def extract_layer8_from_speculative(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
    exit_query_cache: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Tuple, Dict]:
    """
    Modified version of forward_early that extracts and returns layer 8 hidden states
    along with the regular outputs for speculative decoding.
    
    Args:
        model: LLaMA model
        input_ids: Input token IDs
        past_key_values: Optional past key values for faster generation
        exit_query_cache: Optional query cache for exit layer
        
    Returns:
        tuple: (layer_8_hidden_states, past_key_values, exit_query_cache)
    """
    # This implementation assumes your self_speculation.llama_model_utils has a similar 
    # structure to what's shown in the repository files
    
    hidden_states = input_ids.new_zeros(
        (input_ids.shape[0], input_ids.shape[1], model.config.hidden_size)
    ).to(dtype=torch.float16)
    
    # Process embedding layer
    hidden_states = model.model.embed_tokens(input_ids)
    
    # Process layers up to layer 8
    for idx in range(8 + 1):  # +1 to include layer 8
        layer = model.model.layers[idx]
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        
        # If this is the exit layer (layer 8), save the hidden states before processing
        layer_8_hidden_states = None
        if idx == 8:
            layer_8_hidden_states = hidden_states.clone()
            
        # Process the layer
        hidden_states, new_key_value = layer(
            hidden_states=hidden_states,
            attention_mask=None,  # We handle attention mask separately in speculative decoding
            position_ids=torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
        )
        
        # Update past_key_values if caching
        if past_key_values is not None:
            past_key_values[idx] = new_key_value
    
    # Return the hidden states from layer 8, along with updated caches
    return layer_8_hidden_states, past_key_values, exit_query_cache