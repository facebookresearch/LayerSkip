# LayerSkip

## Getting Started
- Clone repo:
```
git clone git@github.com:facebookresearch/LayerSkip.git
```

- Setup environment:
```
conda create --name layer_skip python=3.10
conda activate layer_skip

pip install -r requirements.txt
```

## Run
```
torchrun generate.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
    --data_path dummy \
    --data_format cnn_dm_summarization \
    --num_samples 100 \
    --manifold_output_dir dummy
```

## Evaluate

- Llama 7B continual:
    - CNN/DM Summarization
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_summarization \
            --num_samples 100 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.15924297273159027, 'rouge-1': 0.2301056683063507, 'rouge-2': 0.07174421846866608, 'rouge-3': 0.03521019592881203, 'bleu_score': 0.0, 'exact_match': 251.8300018310547}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 0.9018121194839478}, 'time_per_token': {'mean': 0.02818162873387337}, 'tokens_per_second': {'mean': 35.90497396469116}}
        ```


        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_summarization \
            --num_samples 100 \
            --generation_strategy self_speculative \
            --num_speculations 6 \
            --exit_layer 8 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.15924297273159027, 'rouge-1': 0.22984926402568817, 'rouge-2': 0.07174421846866608, 'rouge-3': 0.03521019592881203, 'bleu_score': 0.0, 'exact_match': 251.83999633789062}, 'acceptance_rate': {'mean': 0.6541042917966843}, 'total_time': {'mean': 0.5943978071212769}, 'time_per_token': {'mean': 0.018574931472539902}, 'tokens_per_second': {'mean': 59.77584112167359}}
        ```

    - CNN/DM Language Modeling
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_lm \
            --num_samples 100 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.04713794216513634, 'rouge-1': 0.06010835990309715, 'rouge-2': 0.019858235493302345, 'rouge-3': 0.008144848048686981, 'bleu_score': 0.0, 'exact_match': 3562.909912109375}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 0.8102598142623901}, 'time_per_token': {'mean': 0.02532061919569969}, 'tokens_per_second': {'mean': 39.88337937355041}}
        ```

        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
            --data_format cnn_dm_lm \
            --num_samples 100 \
            --generation_strategy self_speculative \
            --num_speculations 6 \
            --exit_layer 8 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.14605551958084106, 'rouge-1': 0.24131913483142853, 'rouge-2': 0.048992596566677094, 'rouge-3': 0.018053824082016945, 'bleu_score': 0.0, 'exact_match': 2646.3798828125}, 'acceptance_rate': {'mean': 0.6698643353581428}, 'total_time': {'mean': 6.796451380252838}, 'time_per_token': {'mean': 0.013813270982354879}, 'tokens_per_second': {'mean': 75.60851264953614}}
        ```
    
    - HumanEval
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format human_eval \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.13615505397319794, 'rouge-1': 0.15974164009094238, 'rouge-2': 0.05530855059623718, 'rouge-3': 0.029869550839066505, 'bleu_score': 0.0, 'exact_match': 513.0487670898438}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 5.588762165569678}, 'time_per_token': {'mean': 0.025083595273516526}, 'tokens_per_second': {'mean': 39.88735136171667}}
        ```

        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format human_eval \
            --generation_strategy self_speculative \
            --num_speculations 6 \
            --exit_layer 4 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.13653630018234253, 'rouge-1': 0.16005906462669373, 'rouge-2': 0.05534951388835907, 'rouge-3': 0.02987692505121231, 'bleu_score': 0.0, 'exact_match': 512.4573364257812}, 'acceptance_rate': {'mean': 0.40332102816461063}, 'total_time': {'mean': 3.3100228091565573}, 'time_per_token': {'mean': 0.015546111617146469}, 'tokens_per_second': {'mean': 67.84225034713745}}
        ```

    - CNN/DM Summarization (One Shot)
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format cnn_dm_summarization \
            --n_shot 1 \
            --num_samples 100 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.17083728313446045, 'rouge-1': 0.23529480397701263, 'rouge-2': 0.07809153199195862, 'rouge-3': 0.03981202840805054, 'bleu_score': 0.0, 'exact_match': 236.52000427246094}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 1.0357493662834167}, 'time_per_token': {'mean': 0.03236716769635677}, 'tokens_per_second': {'mean': 31.2203493309021}}
        ```

        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format cnn_dm_summarization \
            --n_shot 1 \
            --num_samples 100 \
            --generation_strategy self_speculative \
            --num_speculations 12 \
            --exit_layer 8 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.17013584077358246, 'rouge-1': 0.23432669043540955, 'rouge-2': 0.07809153199195862, 'rouge-3': 0.03981202840805054, 'bleu_score': 0.0, 'exact_match': 236.6999969482422}, 'acceptance_rate': {'mean': 0.43046684432774784}, 'total_time': {'mean': 0.9261050200462342}, 'time_per_token': {'mean': 0.028940781876444818}, 'tokens_per_second': {'mean': 41.24269124031067}}
        ```

    - XSUM Summarization (Three Shot)
        AR:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format xsum_summarization \
            --n_shot 3 \
            --num_samples 100 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.22860220074653625, 'rouge-1': 0.28382208943367004, 'rouge-2': 0.09564505517482758, 'rouge-3': 0.043253131210803986, 'bleu_score': 0.0, 'exact_match': 100.63999938964844}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 1.2253910517692566}, 'time_per_token': {'mean': 0.03829347036778927}, 'tokens_per_second': {'mean': 26.412611989974977}}
        ```

        SS:
        ```
        torchrun benchmark.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
            --data_path dummy \
            --data_format xsum_summarization \
            --n_shot 3 \
            --num_samples 100 \
            --generation_strategy self_speculative \
            --num_speculations 12 \
            --exit_layer 8 \
            --manifold_output_dir ./logs
        ```
        Result:
        ```
        {'predicted_text': {'rouge-l': 0.228690505027771, 'rouge-1': 0.28392985463142395, 'rouge-2': 0.09567989408969879, 'rouge-3': 0.043278768658638, 'bleu_score': 0.0, 'exact_match': 100.5999984741211}, 'acceptance_rate': {'mean': 0.43856151334941385}, 'total_time': {'mean': 1.176209304332733}, 'time_per_token': {'mean': 0.03675654076039791}, 'tokens_per_second': {'mean': 28.22171961784363}}
        ```

- Llama 1.5B pretrained from scratch:
AR:
```
torchrun benchmark.py --model_path /fsx-atom/melhoushi/xldumps/train_llama2_1.5B_sweep_32_gpus/train_llama2_1.5B_sweep_32_gpus_run008/checkpoints/checkpoint_0050000_consolidated_hf/ \
    --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
    --data_format cnn_dm_summarization \
    --num_samples 100 \
    --manifold_output_dir ./logs
```
Result:
```
{'predicted_text': {'rouge-l': 0.17063584923744202, 'rouge-1': 0.2387697994709015, 'rouge-2': 0.07077603787183762, 'rouge-3': 0.030207907781004906, 'bleu_score': 0.0, 'exact_match': 237.10000610351562}, 'acceptance_rate': {'mean': -1.0}, 'total_time': {'mean': 0.7412997961044312}, 'time_per_token': {'mean': 0.023165618628263475}, 'tokens_per_second': {'mean': 45.00272045135498}}
```

SS:
```
torchrun benchmark.py --model_path /fsx-atom/melhoushi/xldumps/train_llama2_1.5B_sweep_32_gpus/train_llama2_1.5B_sweep_32_gpus_run008/checkpoints/checkpoint_0050000_consolidated_hf/ \
    --data_path /fsx-atom/melhoushi/data/cnn_dm/test.json \
    --data_format cnn_dm_summarization \
    --num_samples 100 \
    --generation_strategy self_speculative \
    --num_speculations 12 \
    --exit_layer 8 \
    --manifold_output_dir ./logs
```
Result:
```
{'predicted_text': {'rouge-l': 0.16813838481903076, 'rouge-1': 0.2398979812860489, 'rouge-2': 0.07607144117355347, 'rouge-3': 0.03642928972840309, 'bleu_score': 0.0, 'exact_match': 250.60000610351562}, 'acceptance_rate': {'mean': 0.019020007958170028}, 'total_time': {'mean': 2.282204785346985}, 'time_per_token': {'mean': 0.07131889954209328}, 'tokens_per_second': {'mean': 14.112926120758056}}
```

## Sweep
- Llama 7B continual:
    - HumanEval
    ```
    torchrun sweep.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
        --data_path dummy \
        --data_format human_eval \
        --generation_strategy self_speculative \
        --num_speculations 6 \
        --exit_layer 4 \
        --num_samples 10 \
        --manifold_output_dir ./logs
    ```
    Result:
    ```
    
    ```


## Correctness
- Llama 7B continual:
    - HumanEval
    ```
    torchrun correctness.py --model_path /fsx-scaling/melhoushi/xldumps/continual_7Bv2_ld_ee_best2/continual_7Bv2_ld_ee_best2_run000/checkpoints/checkpoint_0050000_consolidated_hf/ \
        --data_path dummy \
        --data_format human_eval \
        --generation_strategy self_speculative \
        --num_speculations 6 \
        --exit_layer 4 \
        --num_samples 10 \
        --manifold_output_dir ./logs
    ```
    Result:
    ```
    
    ```