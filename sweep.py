from typing import Dict, List

from arguments import BenchmarkArguments, process_cli_arguments
from benchmark import benchmark, load_model_and_tokenizer, setup
from self_speculation.generator_base import (
    GenerationConfig,
)


def sweep(benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig):
    results: List[Dict] = []
    setup(benchmark_arguments)
    model, tokenizer = load_model_and_tokenizer(benchmark_arguments)
    for exit_layer in range(4, 32, 2):
        for num_speculations in range(4, 14, 2):
            generation_config.exit_layer = exit_layer
            generation_config.num_speculations = num_speculations

            metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config)

            results.append({"exit_layer": exit_layer, "num_speculations": num_speculations, "time_per_token": metric_result['time_per_token']['mean']})
            print(f"exit_layer: {exit_layer}, num_speculations: {num_speculations}, time_per_token: {metric_result['time_per_token']['mean']}")

    print(results)

if __name__ == "__main__":
    benchmark_arguments, generation_config = process_cli_arguments()
    sweep(benchmark_arguments, generation_config)