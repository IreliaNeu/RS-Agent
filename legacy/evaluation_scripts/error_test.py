# import requests
# import json

# response = requests.get(
#   url="https://openrouter.ai/api/v1/auth/key",
#   headers={
#     "Authorization": f"Bearer REDACTED_API_KEY"
#   }
# )

# print(json.dumps(response.json(), indent=2))


import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen3-32B",
    "max_tokens": 512,
    "enable_thinking": False,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "messages": [
        {
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?",
            "role": "user"
        }
    ]
}
headers = {
    "Authorization": "Bearer REDACTED_API_KEY",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())


# import requests
# import json

# response = requests.post(
#   url="https://openrouter.ai/api/v1/chat/completions",
#   headers={
#     "Authorization": "Bearer REDACTED_API_KEY",
#     "Content-Type": "application/json",
#   },
#   data=json.dumps({
#     "model": "openai/gpt-4o-2024-11-20",
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           {
#             "type": "text", 
#             "text": "What is in this image?"
#           },
#           {
#             "type": "image_url",
#             "image_url": {
#               "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#             }
#           }
#         ]
#       }
#     ],
    
#   })
# )
# print(response.json())