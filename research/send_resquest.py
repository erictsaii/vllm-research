import requests
import json
import argparse

def generate_dummy_prompt(length):
    base_word = "hello "
    return base_word * length

def send_completion_request(input_length, model_name, url):
    # 產生 Prompt
    prompt_content = generate_dummy_prompt(input_length)
    
    print(f"[*] Sending request with input length factor: {input_length}")
    print(f"[*] Approximate prompt character length: {len(prompt_content)}")

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "prompt": prompt_content,
        "max_tokens": 32,
        "temperature": 0.0
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # 檢查 HTTP 錯誤
        
        # 解析並列印結果
        result = response.json()
        print("\n[+] Response received:")
        print(json.dumps(result, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\n[!] Request failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send request to LLM with variable input length.")
    
    parser.add_argument(
        "--length", 
        type=int, 
        default=10, 
        help="Number of times to repeat the dummy word to increase input size."
    )
    
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://vllm-proxy:8000/v1/completions", 
        help="Target API URL."
    )

    args = parser.parse_args()

    # MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    # MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

    send_completion_request(args.length, MODEL_NAME, args.url)