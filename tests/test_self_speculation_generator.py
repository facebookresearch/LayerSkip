# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.tests_constants import local_model_path

from self_speculation.generator_base import GenerationConfig
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
import logging

@pytest.fixture
def model_and_config():
    """Fixture to create a model and a generation configuration."""
    # Initialize a lightweight version of LlamaForCausalLM if possible
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        use_safetensors=True,
        device_map="auto",
        torch_dtype=torch.float,
    )

    model.eval()
    config = GenerationConfig(max_steps=4, exit_layer=4, num_speculations=4)

    return model, tokenizer, config

def test_single_step_speculation_handling_eos(model_and_config):
    """Tests the single step speculation and checks EOS handling."""

    model, tokenizer, generation_config = model_and_config
    strategy = SelfSpeculativeGenerationStrategy()
    input_ids = torch.tensor([[tokenizer.encode("my")[1]]], device=model.device)
    input_ids_list = input_ids.tolist()
    
    eos_token_ids = [tokenizer.eos_token_id]
    output_ids = []
    past_key_values = None
    num_speculations = 1

    _, output_ids, _, matches, specs = strategy.single_step_speculation(
        model=model,
        input_ids=input_ids,
        input_ids_list=input_ids_list,
        output_ids=output_ids,
        num_speculations=num_speculations,
        past_key_values=past_key_values,
        eos_token_ids=eos_token_ids,
        calls=0,
        exit_layer=generation_config.exit_layer,
        sample=generation_config.sample,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
    )

    assert matches <= specs

def test_generate_token_ids_with_logit_processors(model_and_config):
    """Test application of logits processors during token generation."""
    model, tokenizer, config = model_and_config
    strategy = SelfSpeculativeGenerationStrategy()
    input_ids = torch.tensor([tokenizer.encode("my")[1], tokenizer.encode("name")[1], tokenizer.encode("is")[1]], device=model.device)
    eos_token_ids = [tokenizer.eos_token_id]
    logits_processor = lambda inputs, logits: torch.log(torch.softmax(logits, dim=-1))

    result = strategy.generate_token_ids(model, input_ids.tolist(), eos_token_ids, config, logits_processors=logits_processor)

    assert len(result.predicted_tokens) > 0
    assert tokenizer.eos_token_id in result.predicted_tokens or len(result.predicted_tokens) == config.max_steps
