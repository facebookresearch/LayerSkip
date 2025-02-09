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
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy

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
    config = GenerationConfig(max_steps=8)

    return model, tokenizer, config


def test_generate_token_ids_with_stopping_criteria(model_and_config):
    """Test stopping criteria application to halt generation early."""
    model, tokenizer, config = model_and_config
    strategy = AutoRegressiveGenerationStrategy()
    input_ids = torch.tensor([tokenizer.encode("my")[1], tokenizer.encode("name")[1], tokenizer.encode("is")[1]], device=model.device)
    eos_token_ids = [tokenizer.eos_token_id]
    stopping_criteria = lambda inputs, scores: torch.tensor([True])  # Stop immediately

    result = strategy.generate_token_ids(model, input_ids.tolist(), eos_token_ids, config, stopping_criteria=stopping_criteria)

    assert len(result.predicted_tokens) == 0 


def test_generate_token_ids_with_logit_processors(model_and_config):
    """Test application of logits processors during token generation."""
    model, tokenizer, config = model_and_config
    strategy = AutoRegressiveGenerationStrategy()
    input_ids = torch.tensor([tokenizer.encode("my")[1], tokenizer.encode("name")[1], tokenizer.encode("is")[1]], device=model.device)
    eos_token_ids = [tokenizer.eos_token_ids]
    logits_processor = lambda inputs, logits: torch.log(torch.softmax(logits, dim=-1))

    result = strategy.generate_token_ids(model, input_ids.tolist(), eos_token_ids, config, logits_processors=logits_processor)

    assert len(result.predicted_tokens) > 0
    assert tokenizer.eos_token_id in result.predicted_tokens or len(result.predicted_tokens) == config.max_steps
