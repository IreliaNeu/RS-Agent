import os
import json

def generate_jsonl_from_folder(image_dir, output_path, exts={".jpg", ".jpeg", ".png"}):
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[-1].lower() in exts
    ]
    image_paths = [os.path.join(image_dir, f).replace("\\", "/") for f in sorted(image_files)]

    with open(output_path, "w", encoding="utf-8") as fout:
        for path in image_paths:
            fout.write(json.dumps({"image_path": path}, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(image_paths)} entries to {output_path}")

if __name__ == "__main__":
    image_dir = "D:/Deeplearning/Change_Agent_main/sampled/loveda_sample_100/images"
    output_jsonl = "D:/Deeplearning/Change_Agent_main/sampled/loveda_input.jsonl"
    generate_jsonl_from_folder(image_dir, output_jsonl)
