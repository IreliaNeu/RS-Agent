import openai  # or other LLM API
import os

def call_llm(prompt: str, model='gpt-4o'):
    # 注意：你需要替换为你自己的API Key调用方式
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response['choices'][0]['message']['content']
