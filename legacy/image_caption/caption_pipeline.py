# 文件路径：Image_Caption/caption_pipeline.py

import os
from tqdm import tqdm
from Image_Caption.Image_caption import ImageCaptionGenerator
from Image_Caption.RefineCaptionsWithEmbedding import CaptionRefiner


def generate_and_refine_pipeline(image_dir: str, output_dir: str, api_key: str) -> str:
    """
    执行从图像目录 -> 生成描述 -> 精选描述（加token） 的完整流程。
    输出：标准格式 JSON 数据集文件路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    caption_output_dir = os.path.join(output_dir, "captions_raw")
    refined_output_dir = os.path.join(output_dir, "refined")

    os.makedirs(caption_output_dir, exist_ok=True)
    os.makedirs(refined_output_dir, exist_ok=True)

    # Step 1: 调用 ImageCaption 生成模块
    print("🚀 Step 1: Generating captions...")
    captioner = ImageCaptionGenerator(api_key)
    captioner.process_folder(image_dir, caption_output_dir)

    # Step 2: 调用 Refine 模块选择最佳描述
    print("🔍 Step 2: Refining captions...")
    refiner = CaptionRefiner(api_key)
    refiner.refine_json_folder(caption_output_dir, refined_output_dir)

    # Step 3: 汇总为标准 multimodal dataset 格式
    print("📦 Step 3: Building dataset...")
    dataset = []
    for fname in os.listdir(refined_output_dir):
        if fname.endswith("_refined.json"):
            with open(os.path.join(refined_output_dir, fname), "r", encoding="utf-8") as f:
                data = f.read()
                try:
                    obj = eval(data) if isinstance(data, str) else data
                except Exception:
                    continue
            dataset.append({
                "image": os.path.join(image_dir, fname.replace("_refined.json", ".png")),
                "caption_tokens": obj.get("final_caption_tokens", obj.get("caption_tokens", []))
            })

    output_json_path = os.path.join(output_dir, "remote_caption_dataset.json")
    import json
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All done. Dataset saved to: {output_json_path}")
    return output_json_path


# ✅ 命令行测试入口（可选）
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Generate and refine image captions for remote sensing images.")
#     parser.add_argument("--api_key", type=str, required=True)
#     parser.add_argument("--image_dir", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     args = parser.parse_args()

#     generate_and_refine_pipeline(args.image_dir, args.output_dir, args.api_key)
