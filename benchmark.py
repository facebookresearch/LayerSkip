import datetime
import os
import random
from dataclasses import dataclass
from typing import Dict
import logging

import torch
import transformers
from tqdm import tqdm

from torchmetrics.text import BLEUScore, ROUGEScore, EditDistance
# TODO: create ExactMatch torchmetrics.text

from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.metric import Metric

from data import get_data, LowercaseProcessingFunction
from utils import ROUGEScoreWrapper

from arguments import BenchmarkArguments, process_cli_arguments
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)

from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy

log = logging.getLogger(__name__)


@dataclass
class EvaluationExample:
    input: str
    output: str


@dataclass
class EvaluationMetrics:
    predicted_text: Dict[str, Metric]
    acceptance_rate: Dict[str, Metric]
    total_time: Dict[str, Metric]
    time_per_token: Dict[str, Metric]
    tokens_per_second: Dict[str, Metric]

    def update(
        self,
        evaluation_example: EvaluationExample,
        generation_result: GenerationResult,
    ) -> None:
        for metric in self.predicted_text.values():
            metric.update(
                evaluation_example.output, generation_result.decoded_prediction
            )

        for metric in self.acceptance_rate.values():
            acceptance_rate = torch.tensor(
                generation_result.generation_strategy_result.acceptance_rate or -1
            )
            metric.update(acceptance_rate)

        for metric in self.total_time.values():
            metric.update(torch.tensor(generation_result.total_time))

        for metric in self.time_per_token.values():
            metric.update(torch.tensor(generation_result.time_per_token))

        for metric in self.tokens_per_second.values():
            metric.update(torch.tensor(generation_result.tokens_per_second))

    def compute(self) -> Dict[str, torch.Tensor]:
        return {
            "predicted_text": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.predicted_text.items()
            },
            "acceptance_rate": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.acceptance_rate.items()
            },
            "total_time": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.total_time.items()
            },
            "time_per_token": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.time_per_token.items()
            },
            "tokens_per_second": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.tokens_per_second.items()
            },
        }

    @classmethod
    def build_metrics(cls) -> "EvaluationMetrics":
        return cls(
            predicted_text={
                "rouge-l": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rougeL",
                        normalizer=LowercaseProcessingFunction,
                    )
                ),
                "rouge-1": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge1", normalizer=LowercaseProcessingFunction
                    )
                ),
                "rouge-2": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge2", normalizer=LowercaseProcessingFunction
                    )
                ),
                "rouge-3": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge3", normalizer=LowercaseProcessingFunction
                    )
                ),
                "bleu_score": BLEUScore(
                    n_gram=4,
                ),
                "exact_match": EditDistance(),
            },
            acceptance_rate={"mean": Mean()},
            total_time={"mean": Mean()},
            time_per_token={"mean": Mean()},
            tokens_per_second={"mean": Mean()},
        )


def setup(benchmark_arguments: BenchmarkArguments):
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=48)
    )
    rank = int(os.environ["LOCAL_RANK"])

    random.seed(benchmark_arguments.seed)
    torch.manual_seed(benchmark_arguments.seed)
    if rank != 0:
        # only run on rank 0, we don't support parallel inference yet
        return


def load_model_and_tokenizer(benchmark_arguments: BenchmarkArguments):
    local_model_path: str = benchmark_arguments.model_path

    # initialize model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        local_model_path, use_fast=False
    )
    config = transformers.LlamaConfig.from_pretrained(local_model_path)
    model = transformers.LlamaForCausalLM.from_pretrained(
        local_model_path,
        config=config,
        torch_dtype=torch.float16,
    )
    model.cuda()
    model.half()
    model.eval()

    return model, tokenizer


def benchmark(
        model: torch.nn.Module, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        benchmark_arguments: BenchmarkArguments, 
        generation_config: GenerationConfig
    ):
    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    else:
        raise Exception(
            f"Unsupported generation strategy: {generation_config.generation_strategy}"
        )

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    evaluation_set = get_data(
        data_path=benchmark_arguments.data_path,
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        data_format=benchmark_arguments.data_format,
        n_shot=benchmark_arguments.n_shot,
        seed=benchmark_arguments.seed,
    )
    metrics = EvaluationMetrics.build_metrics()
    for i, example in enumerate(tqdm(evaluation_set)):
        response: GenerationResult = generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        print(
            f"[Example]: {example.output}\n[Prediction]: {response.decoded_prediction}"
        )
        if response.num_tokens_generated == 0:
            print("Skipping empty generation")
            # TBD: print stats of emprty generations
            continue
        metrics.update(example, response)

    metric_result = metrics.compute()

    return metric_result


def main(benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig):
    setup(benchmark_arguments)
    model, tokenizer = load_model_and_tokenizer(benchmark_arguments)
    metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config)
    print(metric_result)

    # TODO: write to file
    # WorkflowTCRunner.upload_output_to_manifold(
    #     folder_path=benchmark_arguments.manifold_output_dir,
    #     output=metric_result,
    # )


if __name__ == "__main__":
    benchmark_arguments, generation_config = process_cli_arguments()
    main(benchmark_arguments, generation_config)
