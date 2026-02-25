from vllm import LLM, SamplingParams
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MAX_MODEL_LEN = 2048

prompts = [
    "America is a",
    # "The capital of France is",
    # "Hi, how are you?",
]

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name =  "meta-llama/Llama-2-7b-hf"


sampling_params = SamplingParams(temperature=0.6)

llm = LLM(model=model_name,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.95,
            dtype="half",
            enable_chunked_prefill=False)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")