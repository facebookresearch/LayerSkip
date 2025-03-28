# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

import torch
import transformers
from tqdm import tqdm
from data import EvaluationExample

from torchmetrics.text import BLEUScore, ROUGEScore, EditDistance
# TODO: create ExactMatch torchmetrics.text

from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.metric import Metric

from data import get_data, LowercaseProcessingFunction, get_valid_dataset_formats, EvaluationExample
from generate import load_model_and_tokenizer, setup
from utils import ROUGEScoreWrapper

import arguments
from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)

from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.layer_drop_generator import LayerDropGenerationStrategy
from typing import Any
from self_speculation.depth_adaptive_token_generator import DepthAdaptiveTokenGenerationStrategy

log = logging.getLogger(__name__)

@dataclass
class BenchmarkArguments:
    dataset: str
    data_path: Optional[str] = None
    random_shuffle: bool = True
    num_samples: Optional[int] = None
    n_shot: Optional[int] = 0
    template: Optional[str] = None

# @dataclass
# class EvaluationExample:
#     input: str
#     output: str

class ExactMatch(Metric):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)
        self.correct_count = 0
        self.total_count = 0

    def update(self, prediction: str, target: str) -> None:
        self.total_count += 1
        if prediction.strip() == target.strip():  # remove trailing spaces
            self.correct_count += 1

    def compute(self) -> torch.Tensor:
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.device)  # Handle the case where no updates were made
        return torch.tensor(self.correct_count / self.total_count, device=self.device)

    def merge_state(self, metrics: "ExactMatch") -> None:
        """
        Merges the state from another ExactMatch metric instance.
        This is crucial for distributed training/evaluation.
        """
        self.correct_count += metrics.correct_count
        self.total_count += metrics.total_count


class MultipleChoiceExactMatch(Metric):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)
        self.correct_count = 0
        self.total_count = 0

    def update(self, prediction: str, target: str) -> None:
        self.total_count += 1
        
        # Extract answer letter from prediction
        pred_letter = None
        if prediction.strip().startswith(('A', 'B', 'C', 'D')):
            pred_letter = prediction.strip()[0]
        else:
            # Try to find first occurrence of A. B. C. or D.
            for letter in ['A', 'B', 'C', 'D']:
                pattern = letter + '.'
                if pattern in prediction:
                    pred_letter = letter
                    break
        
        # Extract answer letter from target
        target_letter = target.strip()
        if len(target_letter) > 0:
            target_letter = target_letter[0]
        
        print(f"Extracted: prediction={pred_letter}, target={target_letter}")
        
        if pred_letter and target_letter and pred_letter == target_letter:
            self.correct_count += 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")

    def compute(self) -> torch.Tensor:
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.tensor(self.correct_count / self.total_count, device=self.device)

    def merge_state(self, metrics: "MultipleChoiceExactMatch") -> None:
        self.correct_count += metrics.correct_count
        self.total_count += metrics.total_count


class MultipleChoiceAccuracy(Mean):
    def update(self, prediction: str, target: str) -> None:
        # Extract answer letter from prediction
        pred_letter = None
        if prediction.strip().startswith(('A', 'B', 'C', 'D')):
            pred_letter = prediction.strip()[0]
        else:
            # Try to find first occurrence of A. B. C. or D.
            for letter in ['A', 'B', 'C', 'D']:
                pattern = letter + '.'
                if pattern in prediction:
                    pred_letter = letter
                    break
        
        # Extract answer letter from target
        target_letter = target.strip()
        if len(target_letter) > 0:
            target_letter = target_letter[0]
        
        # Pass 1.0 if correct, 0.0 if incorrect
        is_correct = 1.0 if pred_letter and target_letter and pred_letter == target_letter else 0.0
        super().update(torch.tensor(is_correct))


@dataclass
class EvaluationMetrics:
    predicted_text: Dict[str, Metric]
    acceptance_rate: Dict[str, Metric]
    total_time: Dict[str, Metric]
    time_per_token: Dict[str, Metric]
    tokens_per_second: Dict[str, Metric]
    
    def update(
    self,
    evaluation_example: Dict[str, str], 
    generation_result: GenerationResult,
    ) -> None:
        prediction = generation_result.decoded_prediction
        target = evaluation_example["output"]

        # Extract the answer letter from prediction for MMLU
        predicted_letter = None
        if prediction.strip().startswith(('A', 'B', 'C', 'D')):
            predicted_letter = prediction.strip()[0]  # Just take the first character
        else:
            # Try to find first occurrence of A. B. C. or D.
            for letter in ['A', 'B', 'C', 'D']:
                pattern = letter + '.'
                if pattern in prediction:
                    predicted_letter = letter
                    break

        # Extract the answer letter from target for MMLU
        target_letter = target.strip()
        if len(target_letter) > 0:
            target_letter = target_letter[0]  # Just take the first character
        
        print(f"Extracted prediction: {predicted_letter}, Extracted target: {target_letter}")
        
        # Handle the different metrics properly
        for metric_name, metric in self.predicted_text.items():
            if metric_name == "exact_match":
                # ExactMatch expects two arguments
                if predicted_letter and target_letter:
                    metric.update(predicted_letter, target_letter)
                else:
                    metric.update(prediction.strip(), target.strip())
            elif metric_name == "accuracy":
                # Mean metric expects only one argument (the value to average)
                is_correct = 0.0
                if predicted_letter and target_letter and predicted_letter == target_letter:
                    is_correct = 1.0
                metric.update(torch.tensor(is_correct))  # Pass just one value

        # Rest of the updates for other metrics
        for metric in self.acceptance_rate.values():
            if generation_result.generation_strategy_result.acceptance_rate is None:
                acceptance_rate = torch.tensor(0)
            else:
                acceptance_rate = torch.tensor(
                    generation_result.generation_strategy_result.acceptance_rate
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


def benchmark(
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
        benchmark_arguments: BenchmarkArguments,
        generation_config: GenerationConfig,
        seed = None,
    ):
    """
    Benchmark function optimized for multiple-choice QA datasets like MMLU and RACE-M.
    """
    # Set up the appropriate generation strategy.
    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    elif generation_config.generation_strategy == "layerdrop":
        generation_strategy: GenerationStrategy = LayerDropGenerationStrategy(
            dropout_rate=generation_config.dropout_rate,
            seed=generation_config.layerdrop_seed or seed
        )
    elif generation_config.generation_strategy == "depth_adaptive_token":
        generation_strategy: GenerationStrategy = DepthAdaptiveTokenGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
    else:
        raise ValueError(
            f"Unrecognized generation strategy: {generation_config.generation_strategy}"
        )
    multiple_choice_datasets = ["piqa", "hellaswag", "arc_easy", "arc_challenge", 
                               "mmlu", "boolq", "race_m", "race_h", "openbookqa", "winogrande"]
    
    # Determine which benchmark function to use based on dataset type
    if benchmark_arguments.dataset in multiple_choice_datasets:
        # For multiple-choice datasets, optimize generation config
        print(f"Optimizing generation config for multiple-choice dataset: {benchmark_arguments.dataset}")
        
        # Limit token generation to get just the answer
        generation_config.max_steps = min(generation_config.max_steps, 20)
        # Lower temperature for more deterministic responses
        generation_config.temperature = min(generation_config.temperature, 0.3)
        
        # Add stop strings if that attribute exists
        if hasattr(generation_config, 'stop_strings') and not generation_config.stop_strings:
            generation_config.stop_strings = ["\n", "."]
        
        print(f"Updated generation config: max_steps={generation_config.max_steps}, "
              f"temperature={generation_config.temperature}")
              
        # Create custom metrics for multiple-choice QA
        evaluation_metrics = EvaluationMetrics(
            predicted_text={
                "exact_match": MultipleChoiceExactMatch(),
                "accuracy": MultipleChoiceAccuracy(),
            },
            acceptance_rate={"mean": Mean()},
            total_time={"mean": Mean()},
            time_per_token={"mean": Mean()},
            tokens_per_second={"mean": Mean()},
        )

        # initialize generator
        generator = HuggingfaceLlamaGenerator(
            tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
        )

        evaluation_data_points = get_data(
            random_shuffle=benchmark_arguments.random_shuffle,
            num_samples=benchmark_arguments.num_samples,
            dataset=benchmark_arguments.dataset,
            data_path=benchmark_arguments.data_path,
            n_shot=benchmark_arguments.n_shot,
            seed=seed,
            template=benchmark_arguments.template,
        )

        print(f"Benchmarking on {benchmark_arguments.dataset.upper()} with {len(evaluation_data_points)} samples...")

        correct_answers = 0
        total_questions = 0

        for data_point in tqdm(evaluation_data_points, desc=f"Benchmarking {benchmark_arguments.dataset.upper()}"):
            print("Type: ", type(data_point))
            
            if not hasattr(data_point, 'input') or not hasattr(data_point, 'output'):
                print(f"WARNING: Unexpected data point format: {data_point}")
                continue
                
            input_text = data_point.input
            expected_output = data_point.output

            generation_result = generator.generate(
                prompt=input_text,
                generation_config=generation_config,
            )

            predicted_answer = generation_result.decoded_prediction.strip()

            print(f"Input: {input_text}")
            print(f"Predicted: {predicted_answer}")
            print(f"Expected: {expected_output}")

            # Update metrics
            total_questions += 1
            
            # For multiple-choice metrics, we directly pass prediction and target
            for metric_name, metric in evaluation_metrics.predicted_text.items():
                metric.update(predicted_answer, expected_output)
                
            # For other metrics
            for metric in evaluation_metrics.acceptance_rate.values():
                if generation_result.generation_strategy_result.acceptance_rate is None:
                    acceptance_rate = torch.tensor(0)
                else:
                    acceptance_rate = torch.tensor(
                        generation_result.generation_strategy_result.acceptance_rate
                    )
                metric.update(acceptance_rate)

            for metric in evaluation_metrics.total_time.values():
                metric.update(torch.tensor(generation_result.total_time))

            for metric in evaluation_metrics.time_per_token.values():
                metric.update(torch.tensor(generation_result.time_per_token))

            for metric in evaluation_metrics.tokens_per_second.values():
                metric.update(torch.tensor(generation_result.tokens_per_second))

            # Show current progress
            current_metrics = evaluation_metrics.compute()
            print(f"Current Exact Match Accuracy: {current_metrics['predicted_text']['exact_match']:.4f}")
            print(f"Current Mean Accuracy: {current_metrics['predicted_text']['accuracy']:.4f}")
            print("-" * 50)

        final_metrics = evaluation_metrics.compute()

        print(f"\n--- Final Metrics ({benchmark_arguments.dataset.upper()}) ---")
        print(f"Final Exact Match Accuracy: {final_metrics['predicted_text']['exact_match']:.4f}")
        print(f"Final Mean Accuracy: {final_metrics['predicted_text']['accuracy']:.4f}")
        print(f"Total Questions: {total_questions}")

        return final_metrics
    else:
        # Call the original benchmark function with our generation strategy
        return original_benchmark(model, tokenizer, benchmark_arguments, generation_config, seed, generation_strategy)


# Save the original benchmark function for other datasets
def original_benchmark(
        model: torch.nn.Module, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        benchmark_arguments: BenchmarkArguments, 
        generation_config: GenerationConfig,
        seed = None,
        generation_strategy = None,
    ):
    """The original benchmark function for non-multiple-choice datasets."""
    # Use the provided generation strategy or create a new one if not provided
    if generation_strategy is None:
        if generation_config.generation_strategy == "autoregressive":
            generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
        elif generation_config.generation_strategy == "self_speculative":
            generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
        elif generation_config.generation_strategy == "layerdrop":
            generation_strategy: GenerationStrategy = LayerDropGenerationStrategy(
                dropout_rate=generation_config.dropout_rate,
                seed=generation_config.layerdrop_seed or seed
            )
        else:
            raise ValueError(
                f"Unrecognized generation strategy: {generation_config.generation_strategy}"
            )

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    evaluation_set = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        n_shot=benchmark_arguments.n_shot,
        seed=seed,
        data_path=benchmark_arguments.data_path,
        template=benchmark_arguments.template,
    )
    metrics = EvaluationMetrics.build_metrics()
    for i, example in enumerate(tqdm(evaluation_set)):
        response: GenerationResult = generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        print(f"[Prompt]:\n{example.input}")
        print(f"[Reference Response]:\n{example.output}")
        print(f"[Model Response]:\n{response.decoded_prediction}")
        if response.generation_strategy_result.acceptance_rate is not None:
            print(f"[Acceptance Rate]: {response.generation_strategy_result.acceptance_rate}")
        if response.num_tokens_generated == 0:
            print("Skipping metrics of empty generation")
            # TBD: print stats of empty generations
            continue
        metrics.update(example, response)

    metric_result = metrics.compute()

    return metric_result

def main(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log arguments at beginning
    log.info(f"device={device}\n"
             "args={args}\n"
             "benchmark_arguments={benchmark_arguments}\n"
             "generation_config={generation_config}\n"
             "output_fname={output_fname}\n")

    # Setup and Run Benchmark
    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, args.seed)
    print(metric_result)

    # Save config and results to file
    with open(output_fname, "w") as f:
        json.dump(args.__dict__, f)
        json.dump(benchmark_arguments.__dict__, f)
        json.dump(generation_config.__dict__, f)
        json.dump(metric_result, f)

def process_cli_arguments() -> Tuple[arguments.Arguments, BenchmarkArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((arguments.Arguments, BenchmarkArguments, GenerationConfig))
    general_arguments, benchmark_arguments, generation_config = parser.parse_args_into_dataclasses(return_remaining_strings=False)

    assert benchmark_arguments.dataset in get_valid_dataset_formats(), f"{benchmark_arguments.dataset} is not a supported dataset!"
    
    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_arg = {}
        

    return general_arguments, benchmark_arguments, generation_config

if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    log.setLevel(level=logging.INFO) # TODO: set level based on argument
    os.makedirs(args.output_dir, exist_ok=True)
    main(args, benchmark_arguments, generation_config, f"{args.output_dir}/benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# Debug line to add temporarily
print(f"Valid formats: {get_valid_dataset_formats()}")