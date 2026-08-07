import os
import json
import requests
import numpy as np
from typing import List
from tqdm import tqdm

api_url = "https://api.siliconflow.cn/v1/embeddings"
api_key = os.getenv("SILICONFLOW_API_KEY", "")

class CaptionRefiner:
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-Embedding-4B", api_url: str = "https://api.siliconflow.cn/v1/embeddings"):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        payload = {
            "model": self.model,
            "input": texts
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # 提取每个 embedding 向量
        embeddings = [np.array(item["embedding"]) for item in data["data"]]
        return embeddings

    def choose_best_caption(self, captions: List[str]) -> str:
        embeddings = self.get_embeddings(captions)

        # 计算平均相似度（每条与其余两条）
        sims = []
        for i in range(3):
            sim_sum = 0
            for j in range(3):
                if i != j:
                    sim_sum += self.cosine_similarity(embeddings[i], embeddings[j])
            sims.append(sim_sum / 2)

        # 选择平均相似度最高的描述
        best_index = int(np.argmax(sims))
        return captions[best_index]

    def cross_correction_choose(self, captions: List[str]) -> dict:
        assert len(captions) == 3, "Only supports exactly 3 captions."
        embeddings = self.get_embeddings(captions)

        # Step 1: 计算两两相似度
        sim_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(i + 1, 3):
                sim = self.cosine_similarity(embeddings[i], embeddings[j])
                sim_matrix[i][j] = sim_matrix[j][i] = sim

        # Step 2: 找出最不相似的一对
        pairs = [(0, 1), (0, 2), (1, 2)]
        min_pair = min(pairs, key=lambda x: sim_matrix[x[0]][x[1]])

        i, j = min_pair
        k = 3 - i - j  # 剩下的那个
        sim_ki = sim_matrix[k][i]
        sim_kj = sim_matrix[k][j]
        best_index = i if sim_ki > sim_kj else j

        # 可解释输出结构
        explanation = {
            "captions": captions,
            "similarity_matrix": sim_matrix.tolist(),
            "excluded_pair": [i, j],
            "candidate_from_pair": [i, j],
            "comparison_with_remaining": [sim_ki, sim_kj],
            "selected_caption_index": best_index,
            "selected_caption": captions[best_index]
        }
        return explanation


    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def refine_json_file(self, input_json_path: str, output_json_path: str, explanation_log_file: str = None):
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "captions" in data:
            result = self.cross_correction_choose(data["captions"])
        else:
            raise ValueError("JSON must contain a 'captions' field.")

        # 写 refined caption 文件
        caption_text = result["selected_caption"]

        # 去除前缀编号（如“1.”、“2.”、“3.”）
        if caption_text[:2].isdigit() and caption_text[2] in ['.', ')']:
            caption_text = caption_text[3:].strip()
        elif caption_text[:1].isdigit() and caption_text[1] == '.':
            caption_text = caption_text[2:].strip()

        # 转为 token list，并添加特殊符号
        tokens = ["<START>"] + caption_text.lower().split() + ["<END>"]

        final_result = {
            "final_caption_tokens": tokens
        }
        
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Refined: {os.path.basename(output_json_path)} → '{result['selected_caption']}'")
        print("🧠 Explanation:")
        print(f"Similarity matrix: {np.array(result['similarity_matrix'])}")
        print(f"Excluded pair: {result['excluded_pair']}")
        print(f"Compared remaining similarities: {result['comparison_with_remaining']}")
        print(f"→ Selected index: {result['selected_caption_index']}\n")

    # 如果启用解释日志保存
        if explanation_log_file:
            with open(explanation_log_file, "a", encoding="utf-8") as logf:
                logf.write(json.dumps({
                    "file": os.path.basename(input_json_path),
                    "explanation": result
                }, ensure_ascii=False) + "\n")

    def refine_json_folder(self, input_dir: str, output_dir: str, explanation_log: str = None):
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

        for fname in tqdm(files, desc="Refining batch"):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname.replace(".json", "_refined.json"))
            try:
                self.refine_json_file(input_path, output_path, explanation_log_file=explanation_log)
            except Exception as e:
                print(f"[ERROR] Failed on {fname}: {e}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str,default=api_key, help="Your API key for the embedding model")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--explanation_log", type=str, default=None, help="Optional path to save explanation JSONL file")
    args = parser.parse_args()

    refiner = CaptionRefiner(api_key=args.api_key)

    if os.path.isdir(args.input):
        refiner.refine_json_folder(args.input, args.output)
    else:
        refiner.refine_json_file(args.input, args.output)
