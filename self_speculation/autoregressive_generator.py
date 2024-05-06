from typing import List

import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import (
    decode_next_token,
    forward_early,
)


class AutoRegressiveGenerationStrategy(GenerationStrategy):
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
    ) -> GenerationStrategyResult:
        """Variant of `generate` with inputs/outputs formatted as token_ids."""
        past_key_values = None

        input_ids: torch.Tensor = torch.tensor([input_ids]).to("cuda")
        output_ids: List[int] = []

        exit_query_cache = None
        for _ in range(generation_config.max_steps):
            if generation_config.exit_layer > 0:
                model_output = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                )
            else:
                model_output = model(
                    input_ids=input_ids, past_key_values=past_key_values
                )
            logits = model_output.logits
            past_key_values = model_output.past_key_values
            next_token = decode_next_token(logits=logits, token_idx=-1).item()
            if next_token == eos_token_id:
                break
            output_ids.append(next_token)
            # Don't concatenate `next_token` to original `input_ids` since we're using
            # the KV cache (`past_key_values`) to speed up generation.
            input_ids = torch.tensor([[next_token]]).to(input_ids)

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )
