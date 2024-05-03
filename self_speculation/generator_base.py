import time
from dataclasses import dataclass
from typing import List, Optional

import torch

# @manual=fbsource//third-party/pypi/transformers:transformers
import transformers


@dataclass
class GenerationStrategyResult:
    predicted_tokens: List[int]
    acceptance_rate: Optional[float] = None


@dataclass
class GenerationResult:
    generation_strategy_result: GenerationStrategyResult
    decoded_prediction: str
    total_time: float
    time_per_token: float
    tokens_per_second: float


@dataclass
class GenerationConfig:
    max_steps: int = 32
    exit_layer: int = -1
    num_speculations: int = -1
    generation_strategy: str = "autoregressive"


class GenerationStrategy:
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
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

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        example = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        with torch.inference_mode():
            start = time.time()
            generation_strategy_result = self.generation_strategy.generate_token_ids(
                model=self.model,
                input_ids=example["input_ids"].tolist()[0],
                eos_token_id=self.tokenizer.eos_token_id,
                generation_config=generation_config,
            )
            total_time = time.time() - start
        decoded_prediction = self.tokenizer.decode(
            generation_strategy_result.predicted_tokens
        )
        num_tokens_generated = len(generation_strategy_result.predicted_tokens)
        return GenerationResult(
            generation_strategy_result=generation_strategy_result,
            decoded_prediction=decoded_prediction,
            total_time=total_time,
            time_per_token=total_time / num_tokens_generated,
            tokens_per_second=num_tokens_generated / total_time,
        )
