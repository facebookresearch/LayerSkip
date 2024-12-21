# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

import transformers


@dataclass
class GenerationStrategyResult:
    predicted_tokens: List[int]
    acceptance_rate: Optional[float] = None


@dataclass
class GenerationResult:
    generation_strategy_result: GenerationStrategyResult
    decoded_prediction: str
    num_tokens_generated: int
    total_time: float
    time_per_token: float
    tokens_per_second: float


@dataclass
class GenerationConfig:
    max_steps: int = 512
    exit_layer: int = -1
    num_speculations: int = -1
    generation_strategy: str = "autoregressive"
    sample: bool = True
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 0.9
    no_repeat_ngram_size: int = None
    stop_words: List[str] = None
    stop_token_ids: List[int] = None

    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []

class GenerationStrategy:
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
        raise NotImplementedError()


class HuggingfaceLlamaGenerator:
    def __init__(
        self,
        tokenizer: transformers.LlamaTokenizer,
        model: transformers.LlamaForCausalLM,
        generation_strategy: GenerationStrategy,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.generation_strategy = generation_strategy

    def create_logits_processors(
            self,
            generation_config: GenerationConfig,
    ) -> transformers.generation.logits_process.LogitsProcessorList:
        logits_processors: transformers.generation.logits_process.LogitsProcessorList = transformers.generation.logits_process.LogitsProcessorList()
        if generation_config.no_repeat_ngram_size:
            logits_processors.append(transformers.generation.logits_process.NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))

        return logits_processors

    def create_stopping_criteria(
            self,
            generation_config: GenerationConfig,
    ) -> transformers.StoppingCriteriaList:
        stopping_criteria: transformers.StoppingCriteriaList = transformers.StoppingCriteriaList()
        if generation_config.stop_words:
            stopping_criteria.append(transformers.StopStringCriteria(self.tokenizer, generation_config.stop_words))

        return stopping_criteria

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationResult:
        example = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        logits_processors = self.create_logits_processors(generation_config=generation_config)
        stopping_criteria = self.create_stopping_criteria(generation_config)
        eos_token_ids = generation_config.stop_token_ids + [self.tokenizer.eos_token_id]
        with torch.inference_mode():
            start = time.time()
            generation_strategy_result = self.generation_strategy.generate_token_ids(
                model=self.model,
                input_ids=example["input_ids"].tolist()[0],
                eos_token_ids=eos_token_ids,
                generation_config=generation_config,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            total_time = time.time() - start
        decoded_prediction = self.tokenizer.decode(
            generation_strategy_result.predicted_tokens
        )
        num_tokens_generated = len(generation_strategy_result.predicted_tokens)
        return GenerationResult(
            generation_strategy_result=generation_strategy_result,
            decoded_prediction=decoded_prediction,
            num_tokens_generated=num_tokens_generated,
            total_time=total_time,
            time_per_token=total_time / num_tokens_generated if num_tokens_generated > 0 else None,
            tokens_per_second=num_tokens_generated / total_time,
        )
