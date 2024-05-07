from typing import Dict, List
import pandas as pd
from datetime import datetime
import os
import tabulate

from arguments import BenchmarkArguments, process_cli_arguments
from benchmark import benchmark, load_model_and_tokenizer, setup
from self_speculation.generator_base import (
    GenerationConfig,
)


def sweep(benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    results: List[Dict] = []
    setup(benchmark_arguments)
    model, tokenizer = load_model_and_tokenizer(benchmark_arguments)
    for exit_layer in range(4, 20, 10):
        for num_speculations in range(4, 12, 8):
            generation_config.exit_layer = exit_layer
            generation_config.num_speculations = num_speculations

            metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config)

            results.append({"exit_layer": exit_layer, "num_speculations": num_speculations, "time_per_token": metric_result['time_per_token']['mean']})
            df = pd.DataFrame(results) 
            # Update table every iteration
            df.to_csv(output_fname, index=False)
            print(f"exit_layer: {exit_layer}, num_speculations: {num_speculations}, time_per_token: {metric_result['time_per_token']['mean']}")

    # Print summary table
    print("\n")
    header = results[0].keys()
    rows =  [x.values() for x in results]
    print(tabulate.tabulate(rows, header))

if __name__ == "__main__":
    benchmark_arguments, generation_config = process_cli_arguments()
    os.makedirs("./logs/", exist_ok=True)
    sweep(benchmark_arguments, generation_config, f"./logs/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")