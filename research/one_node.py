# SPDX-License-Identifier: Apache-2.0

import os
import time
from multiprocessing import Event, Process

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name =  "meta-llama/Llama-2-7b-hf"

prompts = [
    "America is a",
    # "America is a",
    "The capital of France is",
    # "Hi, how are you?",
]

MAX_MODEL_LEN = 2048

def run_prefill(prefill_done):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # ktc = KVTransferConfig.from_cli(
    #     '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
    # )
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
    )

    llm = LLM(model=model_name,
              kv_transfer_config=ktc,
              max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.95,
              dtype="half")

    # prefill_start_time = time.time()
    llm.generate(prompts, sampling_params)
    # prefill_end_time = time.time()

    # prefill_duration = prefill_end_time - prefill_start_time
    # print(f"Prefill duration: {prefill_duration:.2f} seconds")

    # Share the exact time prefill finished
    prefill_done.set()

    # global _KV_TRANSFER
    # _KV_TRANSFER.close()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")
    # finally:
    #     from vllm.distributed import parallel_state
    #     if getattr(parallel_state, "_KV_TRANSFER", None) is not None:
    #         print("[test.py] Closing _KV_TRANSFER agent (prefill).")
    #         parallel_state._KV_TRANSFER.close()
    #         parallel_state._KV_TRANSFER = None


def run_decode(prefill_done):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0)

    # ktc = KVTransferConfig.from_cli(
    #     '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
    # )
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
    )

    decode_init_start_time = time.time()
    llm = LLM(model=model_name,
              kv_transfer_config=ktc,
              max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.95,
              dtype="half")

    print("Waiting for prefill node to finish...")
    prefill_done.wait()

    # Time when decode starts generating (after KV cache is available)
    # decode_generate_start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    # decode_end_time = time.time()

    # Calculate durations
    # decode_generate_duration = decode_end_time - decode_generate_start_time

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


    # print(f"Decode duration: {decode_generate_duration:.2f} seconds")
    
    # Cleanup KV_TRANSFER
    # from vllm.distributed import parallel_state
    # if getattr(parallel_state, "_KV_TRANSFER", None) is not None:
    #     print("[test.py] Closing _KV_TRANSFER agent (decode).")
    #     parallel_state._KV_TRANSFER.close()
    #     parallel_state._KV_TRANSFER = None


if __name__ == "__main__":
    prefill_done = Event()

    prefill_process = Process(target=run_prefill, args=(prefill_done,))
    decode_process = Process(target=run_decode, args=(prefill_done,))

    prefill_process.start()
    decode_process.start()

    decode_process.join()
    prefill_process.terminate()