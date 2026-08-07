import os
import json
import base64
import requests
from tqdm import tqdm
from typing import List, Dict

api_url = "https://api.siliconflow.cn/v1/chat/completions"
api_key = os.getenv("SILICONFLOW_API_KEY", "")


class ImageCaptionGenerator:
    def __init__(self, api_key: str):
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.api_key = api_key
        self.model = "Qwen/Qwen2.5-VL-72B-Instruct"
        self.prompt = (
            "You are a remote sensing image analysis expert. Please describe this image in detail. "
            "Identify all objects, their approximate quantities, and their relative positions "
            "(e.g., top-left, center, bottom-right). Generate three distinct, concise descriptions, "
            "each in a single sentence. Number each line as 1., 2., and 3."
        )

    def encode_image_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"  # 或 jpg 视图像格式而定

    def generate_captions(self, image_path: str) -> List[str]:
        image_base64_url = self.encode_image_base64(image_path)

        payload = {
            "model": self.model,
            "stream": False,
            "max_tokens": 512,
            "enable_thinking": True,
            "thinking_budget": 4096,
            "min_p": 0.05,
            "temperature": 1.0,
            "top_p": 0.7,
            "top_k": 100,
            "frequency_penalty": 0.8,
            "n": 1,
            "stop": [],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64_url}
                        },
                        {
                            "type": "text",
                            "text": self.prompt
                        }
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        return [line.strip() for line in text.strip().split('\n') if line.strip()]

    def process_folder(self, image_dir: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))]

        for fname in tqdm(image_files, desc="Generating captions"):
            image_path = os.path.join(image_dir, fname)
            try:
                captions = self.generate_captions(image_path)
                output_data = {
                    # "image": os.path.relpath(image_path, start=output_dir),
                    "captions": captions
                }

                # 保存为与图像同名的 .json 文件
                json_name = os.path.splitext(fname)[0] + ".json"
                json_path = os.path.join(output_dir, json_name)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"[ERROR] Failed to process {fname}: {e}")

        print(f"\n✅ All done! {len(image_files)} JSON files saved to: {output_dir}")



# ✅ 可独立运行
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch captioning of remote sensing images using Qwen-VL.")
    parser.add_argument("--api_key", type=str,default=api_key, help="Your DashScope API Key")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, default="captions.json", help="Output JSON file path")

    args = parser.parse_args()

    captioner = ImageCaptionGenerator(api_key=args.api_key)
    captioner.process_folder(args.image_dir, args.output)  