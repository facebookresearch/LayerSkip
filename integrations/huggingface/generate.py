import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

accelerator = Accelerator()


checkpoint = "melhoushi/layerskip-llama2-7b-topv1-v2" # "melhoushi/layerskip-llama3.2-1b-topv1-v7"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.add_eos_token=False

prompt = "### Instruction: Set alarm for 6am every day\n ### Response: "
inputs = tokenizer(prompt, return_tensors="pt").to(device)

do_sample = False

model.generation_config.pad_token_id = tokenizer.pad_token_id

warmup=4
for _ in range(warmup):
    outputs = model.generate(**inputs, max_new_tokens=40)

start=time.time()
outputs = model.generate(**inputs, do_sample=do_sample, max_new_tokens=40)
delta_time_orig=time.time()-start
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
print(f"Orig Time: {delta_time_orig}")

for i in range(1, 16):
    start=time.time()
    outputs = model.generate(**inputs, assistant_early_exit=i, do_sample=do_sample, max_new_tokens=40)
    delta_time_skip=time.time()-start
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    print(f"Layerskip Time: {delta_time_skip}")

    speedup =  delta_time_orig / delta_time_skip
    print(f"For Layer: {i} Speedup: {speedup}")
