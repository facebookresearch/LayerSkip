# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import transformers
import math

@dataclass
class ForwardResult:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    exit_query_cache: Optional[List[torch.Tensor]] = None

# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = transformers.generation.logits_process.TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = transformers.generation.logits_process.TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits

def decode_next_token(
    logits: torch.Tensor,
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> torch.Tensor:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        return next_token, None
    else:
        if not token_idx:
            logits.squeeze_(dim=0)
        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
            next_token.transpose_(1, 0)
        return next_token, probabilities


def crop_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    maximum_length: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(past_key_values)):
        if past_key_values[idx] is None or past_key_values[idx][0] == [] or past_key_values[idx][0] is None:
            break
        new_past.append(
            (
                past_key_values[idx][0][:, :, :maximum_length, :],
                past_key_values[idx][1][:, :, :maximum_length, :],
            )
        )
    past_key_values = tuple(new_past)
    return past_key_values


# Our forward_early(...) and forward_remainder(...) functions currently use transformers library's legacy KV cache implementation that is less efficient.
# To ensure an apples to apples comparison, we created this forward function to use in autoregressive decoding to ensure it uses the same KV cache implementation instead.
# FIXME: update forward_early(...) and forward_remainder(...) to use the updated more efficient KV cache implementation.
def forward(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    for decoder_layer in model.model.layers:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_key_values = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(
        logits=logits, past_key_values=past_key_values
    )


# TODO: update forward_early(...) to use transformers' new KV cache implementation rather than legacy.
def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    for decoder_layer in model.model.layers[:exit_layer]:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_key_values = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)
    return ForwardResult(
        logits=logits, past_key_values=past_key_values, exit_query_cache=exit_query_cache
    )


# TODO: update forward_remainder(...) to use transformers' new KV cache implementation rather than legacy.
def forward_remainder(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    num_tokens_to_generate: int = 1
    seq_length_with_past = seq_length
    draft_past_key_values_length: int = 0
    full_past_key_values_length: int = 0

    if past_key_values is not None and past_key_values[0] is not None:
        # it's okay to use the first layer because the draft model necessairly computes it
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        # the total sequence length is the past key values since that includes the draft tokens

        # the last layer should not have been skipped, we can get this to check how many of the tokens have gone through full
        # verification
        if len(past_key_values) == len(model.model.layers):
            full_past_key_values_length = past_key_values[-1][0].shape[2]
        else:
            # we have not done a full pass yet so the history is 0
            full_past_key_values_length = 0

        seq_length_with_past = num_tokens_to_generate + draft_past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    inputs_embeds = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length_with_past,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    early_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, num_tokens_to_generate),
        inputs_embeds,
        draft_past_key_values_length,
    )

    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,  # we have no past for the full model
    )

    next_decoder_cache = []
    hidden_states = inputs_embeds
    # TODO simplify
    full_hidden_states: Optional[torch.FloatTensor] = None
    for idx, decoder_layer in enumerate(model.model.layers):
        is_early_exit = idx < exit_layer
        past_key_value = (
            past_key_values[idx]
            if (past_key_values is not None and idx < len(past_key_values))
            else None
        )
        if is_early_exit:
            # early hidden states: B x num_gen x C
            early_hidden_states = hidden_states[:, -num_tokens_to_generate:]
            early_position_ids = position_ids[:, -num_tokens_to_generate:]
            hidden_states, past_key_values = decoder_layer(
                early_hidden_states,
                attention_mask=early_attention_mask,
                position_ids=early_position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
        else:
            if full_hidden_states is None and exit_query_cache is not None:
                # first time seeing the full hidden states, we need to rely on the
                # query cache
                # only use if exit query cache exists, if not this is our first call
                full_hidden_states = torch.cat(
                    [exit_query_cache, hidden_states[:, -num_tokens_to_generate:]],
                    dim=1,
                )
            else:
                # we already have seen the fully hidden states we can re-use them now
                full_hidden_states = hidden_states
            hidden_states, past_key_values = decoder_layer(
                full_hidden_states,
                attention_mask=full_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )

    past_key_values = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(
        logits=logits, past_key_values=past_key_values, exit_query_cache=exit_query_cache
    )

def forward_with_layerdrop(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    dropout_rate: float = 0.2,
    time_step: int = 0,
    max_time_steps: int = 1,
    seed: Optional[int] = None,
) -> ForwardResult:
    """Forward pass with LayerDrop: randomly dropping layers based on the paper's formula.
    
    Args:
        model: LlamaForCausalLM model
        input_ids: Input token ids
        past_key_values: KV cache from previous forward passes
        dropout_rate: Maximum dropout rate (pmax in the paper)
        time_step: Current iteration/time step for curriculum
        max_time_steps: Total number of time steps for curriculum scaling
        seed: Random seed for reproducibility
    """
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Setup position ids and attention mask (similar to forward())
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Initial embeddings
    hidden_states = inputs_embeds
    
    # Calculate time-dependent scaling S(t)
    # For fine-tuning or inference, we use S(t) = 1
    # For pre-training, we would use the exponential curriculum
    if max_time_steps > 1:
        # Exponential curriculum
        s_t = math.exp(time_step * math.log(2) / (max_time_steps - 1)) - 1
    else:
        s_t = 1.0
    
    # Process through layers with LayerDrop
    total_layers = len(model.model.layers)
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        # Calculate per-layer dropout scaling D(l)
        d_l = math.exp(layer_idx * math.log(2) / (total_layers - 1)) - 1
        
        # Calculate dropout probability for this layer
        layer_dropout_prob = s_t * d_l * dropout_rate
        
        # Bernoulli mask: 1 = keep layer, 0 = drop layer
        if torch.rand(1).item() < layer_dropout_prob:
            # Skip this layer
            continue
            
        # Process layer normally
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    # Final norm and head
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(
        logits=logits, 
        past_key_values=past_key_values.to_legacy_cache()
    )

def forward_depth_adaptive_token(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    halting_threshold: float = 0.99,
    min_layers: int = 4,
    max_layers: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Token-level forward pass with early exit.
    
    For each token, computes a halting probability at every layer:
      - h_t^(l) = sigmoid(w^T * h_t^(l) + b)
    Accumulates these probabilities per token until reaching a threshold.
    Returns:
      - final_hidden_states: A weighted combination of hidden states per token.
      - exit_mask: A boolean mask indicating which tokens have halted.
    """
    device = input_ids.device
    hidden_states = model.model.embed_tokens(input_ids)  # Shape: [batch, seq_len, hidden_dim]
    batch_size, seq_len, hidden_dim = hidden_states.shape
    # Initialize final outputs and per-token cumulative halting probability.
    final_outputs = torch.zeros_like(hidden_states)
    cumulative_halting_prob = torch.zeros(batch_size, seq_len, device=device)
    exit_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    total_layers = len(model.model.layers)
    max_layers = max_layers or total_layers

    # Initialize halting module (a simple linear layer) once.
    halting_layer = torch.nn.Linear(hidden_dim, 1).to(device)

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        if layer_idx >= max_layers:
            break
        # Process through the current layer.
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,      # Adjust based on your model specifics.
            position_ids=None,        # Adjust as needed.
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        # Compute per-token halting probabilities.
        # h_t^(l) = sigmoid(halting_layer(hidden_states)) → shape: [batch, seq_len, 1]
        halting_probs = torch.sigmoid(halting_layer(hidden_states)).squeeze(-1)  # Shape: [batch, seq_len]
        # Calculate remaining probability for each token.
        remaining_prob = 1 - cumulative_halting_prob
        # Current contribution: current_halting = h_t^(l) * remaining_prob.
        current_halting = halting_probs * remaining_prob
        cumulative_halting_prob += current_halting
        # Weight the hidden states by the current halting probability.
        final_outputs += hidden_states * current_halting.unsqueeze(-1)

        # Enforce minimum layers before considering halting.
        if layer_idx < min_layers - 1:
            continue

        # Update exit mask where cumulative probability meets or exceeds threshold.
        newly_halted = cumulative_halting_prob >= halting_threshold
        exit_mask = exit_mask | newly_halted

        # If all tokens have halted, we can break early.
        if exit_mask.all():
            break

    # For tokens that never reached the threshold, add remaining hidden states weighted by residual.
    final_outputs += hidden_states * (1 - cumulative_halting_prob).unsqueeze(-1)
    return final_outputs, exit_mask