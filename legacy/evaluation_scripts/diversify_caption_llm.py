import os
import json
import time
import argparse
import re
import requests
from tqdm import tqdm

def call_llm_openai_style(prompt, model, api_key, api_url, max_retry=3):
    """兼容 OpenAI / OpenRouter / DashScope 等 OpenAI 风格接口"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    for attempt in range(max_retry):
        try:
            response = requests.post(api_url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            resp_json = response.json()

            # 通义千问兼容
            if "output" in resp_json:
                return resp_json["output"]["text"]
            elif "choices" in resp_json:
                return resp_json["choices"][0]["message"]["content"]
            else:
                raise ValueError("Unknown response format.")

        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(1)

    return "[]"

def parse_augmented_response(text):
    """将 LLM 输出处理为列表格式"""
    try:
        if text.strip().startswith("["):
            return json.loads(text)
        else:
            lines = text.split("\n")
            return [line.lstrip("1234567890.- ").strip() for line in lines if line.strip()]
    except Exception:
        return [text.strip()]

def make_prompt(original_text, n_versions):
    return (
        f"Please rewrite the following change description and generate {n_versions} versions "
        f"with different styles and diverse language, but consistent meaning. "
        f"This is used for remote sensing image change detection task.\n\n"
        f"{original_text}\n\nReturn all versions as a Python list."
    )

def diversify_batch(args):
    input_dir = args.input_dir
    output_dir = os.path.join(args.output_base, args.model_name, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
    for fname in tqdm(files):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname.replace(".txt", ".json"))

        if os.path.exists(output_path):
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            original = f.readline().strip()
        if not original:
            continue

        prompt = make_prompt(original, args.n_versions)
        response = call_llm_openai_style(prompt, args.model_name, args.api_key, args.api_url)
        augmented = parse_augmented_response(response)

        if args.output_format == "list-token":
            out_data = [["<START>"] + x.split() + ["<END>"] for x in augmented]
        else:
            out_data = {
                "original": original,
                "augmented": augmented
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote Sensing Caption Diversifier")
    parser.add_argument("--input_dir", required=True, help="原始描述所在目录，每个文件首行为描述")
    parser.add_argument("--output_base", default="./outputs", help="输出根目录（自动根据模型+数据集命名）")
    parser.add_argument("--model_name", required=True, help="LLM 模型名，如 gpt-4o、qwen-turbo")
    parser.add_argument("--api_key", required=True, help="API Key")
    parser.add_argument("--api_url", required=True, help="API URL（兼容 OpenAI 风格）")
    parser.add_argument("--dataset_name", required=True, help="数据集名称，如 LEVIR-CD、WHU-CD")
    parser.add_argument("--n_versions", type=int, default=3, help="每条描述生成几条多样化版本")
    parser.add_argument("--output_format", choices=["json", "list-token"], default="list-token")

    args = parser.parse_args()
    diversify_batch(args)
