import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

# Try different models
models_to_test = [
    "meta/llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "nvidia/nemotron-mini-4b-instruct"
]

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

for model in models_to_test:
    print(f"\nTesting model: {model}")
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 20
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"✓ WORKS! Use this model: {model}")
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
        break
    else:
        print(f"✗ Failed: {response.text[:100]}")