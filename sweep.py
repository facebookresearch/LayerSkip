from typing import Dict, List, Tuple
import pandas as pd
import transformers
from datetime import datetime
import os
import tabulate
import torch

from arguments import Arguments, simple_parse_args_string
from benchmark import benchmark, load_model_and_tokenizer, process_cli_arguments, setup, BenchmarkArguments
from self_speculation.generator_base import (
    GenerationConfig,
)

def sweep(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    results: List[Dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    for exit_layer in range(4, len(model.model.layers), 2):
        for num_speculations in range(4, 12, 2):
            generation_config.exit_layer = exit_layer
            generation_config.num_speculations = num_speculations

            metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, args.seed)

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
    args, benchmark_arguments, generation_config = process_cli_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    sweep(args, benchmark_arguments, generation_config, f"{args.output_dir}/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")