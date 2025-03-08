import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "phi",  # Change to your model
    "prompt": "What is AI?",
    "stream": False   # Set to True for streamed responses
}

response = requests.post(url, json=payload)
print(response.json()['response'])
