# Implementation of a DeBERTa-based adapter for enhancing LLaMA intermediate representations

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import transformers
from transformers import DebertaV2Model, DebertaV2Config

class DebertaAdapter(nn.Module):
    """
    Adapter that uses DeBERTa v3 to enhance intermediate layer representations from LLaMA.
    The adapter projects LLaMA hidden states to DeBERTa's input space, processes them 
    through DeBERTa, and then projects back to LLaMA's vocabulary space.
    """
    
    def __init__(
        self, 
        llama_hidden_size: int,
        llama_vocab_size: int,
        deberta_hidden_size: int = 768,
        deberta_num_layers: int = 2,
        deberta_num_attention_heads: int = 12,
        dropout_prob: float = 0.1
    ):
        """
        Initialize the DeBERTa adapter.
        
        Args:
            llama_hidden_size: Hidden size of LLaMA model
            llama_vocab_size: Vocabulary size of LLaMA model
            deberta_hidden_size: Hidden size for DeBERTa model
            deberta_num_layers: Number of layers for DeBERTa model
            deberta_num_attention_heads: Number of attention heads for DeBERTa model
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        # Input projection: LLaMA hidden size -> DeBERTa hidden size
        self.input_projection = nn.Linear(llama_hidden_size, deberta_hidden_size)
        
        # Initialize a small DeBERTa model
        self.deberta_config = DebertaV2Config(
            hidden_size=deberta_hidden_size,
            num_hidden_layers=deberta_num_layers,
            num_attention_heads=deberta_num_attention_heads,
            intermediate_size=4 * deberta_hidden_size,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
        )
        self.deberta = DebertaV2Model(self.deberta_config)
        
        # Output projection: DeBERTa hidden size -> LLaMA vocab size
        self.output_projection = nn.Linear(deberta_hidden_size, llama_vocab_size)
        
        # Layer normalization for better stability
        self.input_layernorm = nn.LayerNorm(deberta_hidden_size)
        self.output_layernorm = nn.LayerNorm(llama_vocab_size)
        
    def forward(self, hidden_states: torch.Tensor,attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process LLaMA hidden states through the DeBERTa adapter.
        
        Args:
            hidden_states: Hidden states from LLaMA layer 8
                        Shape: [batch_size, llama_hidden_size] or [batch_size, seq_len, llama_hidden_size]
            attention_mask: Optional attention mask
                
        Returns:
            Logits over LLaMA vocabulary
        """
        # Fix dimension issue - add sequence dimension if it's missing
        if len(hidden_states.shape) == 2:
            # If shape is [batch_size, hidden_size], add a sequence dimension
            hidden_states = hidden_states.unsqueeze(1)  # Now [batch_size, 1, hidden_size]
        
        # Project to DeBERTa hidden size
        projected_hidden_states = self.input_projection(hidden_states)
        projected_hidden_states = self.input_layernorm(projected_hidden_states)
        
        # Create default attention mask based on the sequence length
        batch_size, seq_length = projected_hidden_states.size()[:2]
        device = projected_hidden_states.device
        
        if attention_mask is None:
            # Create a default attention mask (all 1s)
            attention_mask = torch.ones(batch_size, seq_length, device=device)
        
        # Convert to the format expected by DeBERTa
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Process through DeBERTa
        outputs = self.deberta(
            inputs_embeds=projected_hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        
        # Project DeBERTa outputs to LLaMA vocab space
        deberta_hidden_states = outputs.last_hidden_state
        logits = self.output_projection(deberta_hidden_states)
        logits = self.output_layernorm(logits)
        
        # If we added a sequence dimension earlier, remove it now
        if len(hidden_states.shape) == 3 and hidden_states.shape[1] == 1:
            logits = logits.squeeze(1)
        
        return logits

    
    def save(self, path: str):
        """Save the adapter model to the specified path."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, **init_kwargs):
        """Load the adapter model from the specified path."""
        adapter = cls(**init_kwargs)
        adapter.load_state_dict(torch.load(path))
        return adapter