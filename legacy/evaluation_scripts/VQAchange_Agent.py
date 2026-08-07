#!/usr/bin/env python3
"""
LLM-as-Judge pipeline for evaluating remote sensing image change descriptions (RSVQA).

Input: Each line is {"image_path_A": "xxx_before.jpg", "image_path_B": "xxx_after.jpg"}
Workflow:
1. Use multiple VLLMs to generate change descriptions for the same image pair.
2. GPT-4o selects the best description and scores all.
3. Save results to output.
"""

import argparse
import json
import time
import requests
import sys
import re
import base64
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-Judge pipeline for RSVQA (Change Detection)")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL with {'image_path_A': str, 'image_path_B': str}")
    parser.add_argument("--llm_config", "-c", required=True, help="OpenRouter LLM config JSON")
    parser.add_argument("--mapping_output", "-m", required=True, help="LLM label map output path")
    parser.add_argument("--output", "-o", required=True, help="Full result JSONL")
    parser.add_argument("--best_output", "-b", required=True, help="Best captions JSONL")

    # GPT-4o Judge
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument("--judge_api_key", required=True)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_max_tokens", type=int, default=5)

    # scoring
    parser.add_argument("--score_temperature", type=float, default=0.7)
    parser.add_argument("--score_max_tokens", type=int, default=60)

    parser.add_argument("--delay", type=float, default=1.0)
    return parser.parse_args()


def load_llm_configs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["llms"]


def encode_image_as_base64(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode("utf-8")
    return encoded


def call_openrouter(llm, messages, temperature, max_tokens, max_retries=8, retry_delay=2.0):
    """
    Call a single OpenRouter LLM endpoint with automatic retries on failure.
    """
    url = llm["api_base"].rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm['api_key']}"
    }
    payload = {
        "model": llm["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[⚠️ Attempt {attempt}/{max_retries}] Error calling {llm['name']}: {e}")
            if attempt < max_retries:
                sleep_time = retry_delay + random.uniform(0, 1.0)
                print(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                return f"Error: {e}"


def generate_caption_from_images(image_path_A, image_path_B, llm, args):
    """Use one VLLM to describe the changes between two images."""
    prompt = (
        "<think> You are a professional remote sensing imagery analysis assistant."
        " Your task is to compare two remote sensing images (before and after) and describe the land surface changes."
        " First, carefully analyze both images to identify differences (e.g., land cover, urban expansion, disaster impact)."
        " Then, generate a precise change description with scientific accuracy."
        " Respond in JSON format with two fields: 'reasoning' and 'change description'.</think>"
    )
    imgA_b64 = encode_image_as_base64(image_path_A)
    imgB_b64 = encode_image_as_base64(image_path_B)

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here are two remote sensing images for comparison (before and after):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgA_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgB_b64}"}}
            ]
        }
    ]

    return call_openrouter(
        llm, messages,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens * 40
    )


def judge_best_option(options, args, image_path_A, image_path_B, max_retries=5, retry_delay=2.0):
    """Ask GPT-4o to choose the best caption for the change description."""
    url = "http://chatapi.littlewheat.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.judge_api_key}",
        "Content-Type": "application/json"
    }

    imgA_b64 = encode_image_as_base64(image_path_A)
    imgB_b64 = encode_image_as_base64(image_path_B)
    system = (
        "As a remote sensing expert, select the most accurate change caption for these two images.\n"
        "Focus on:\n"
        "1. Correct identification of changes\n"
        "2. Relevant details\n"
        "3. No false information\n"
        "Reply ONLY with the letter (A/B/C...) of the best option."
    )
    choices_text = "\n".join(f"{k}. {v}" for k, v in options.items())

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "We have multiple remote sensing change captions for these images:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgA_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgB_b64}"}},
                {"type": "text", "text": f"\n\n{choices_text}\n\nWhich one is the best? Return only the single letter."}
            ]
        }
    ]

    payload = {
        "model": args.judge_model,
        "messages": messages,
        "temperature": args.judge_temperature,
        "max_tokens": args.judge_max_tokens,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip().upper()
        except Exception as e:
            print(f"[⚠️ Judge Attempt {attempt}/{max_retries}] {image_path_A},{image_path_B}: {e}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return None


def score_options(options, args, image_path_A, image_path_B, max_retries=5, retry_delay=2.0):
    """Use GPT-4o to rate each caption 1–10 for informativeness and clarity."""
    url = "http://chatapi.littlewheat.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.judge_api_key}",
        "Content-Type": "application/json"
    }

    imgA_b64 = encode_image_as_base64(image_path_A)
    imgB_b64 = encode_image_as_base64(image_path_B)
    system = "You are a remote sensing expert scoring change captions for informativeness and clarity."
    choices_text = "\n".join(f"{k}. {v}" for k, v in options.items())

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please score these captions for the following image pair:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgA_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgB_b64}"}},
                {"type": "text", "text": f"{choices_text}\n\nScore each caption from 1 (worst) to 10 (best). Return JSON like {{\"A\":9, \"B\":7, ...}}"}
            ]
        }
    ]

    payload = {
        "model": args.judge_model,
        "messages": messages,
        "temperature": args.score_temperature,
        "max_tokens": args.score_max_tokens,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r"\{[\s\S]*\}", content)
            return json.loads(match.group()) if match else {k: None for k in options}
        except Exception as e:
            print(f"[⚠️ Score Attempt {attempt}/{max_retries}] {image_path_A},{image_path_B}: {e}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return {k: None for k in options}


def main():
    args = parse_args()
    llms = load_llm_configs(args.llm_config)
    labels = [chr(ord("A") + i) for i in range(len(llms))]
    mapping = {label: llm["name"] for label, llm in zip(labels, llms)}
    Path(args.mapping_output).write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    fout_all = open(args.output, "w", encoding="utf-8")
    fout_best = open(args.best_output, "w", encoding="utf-8")

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            image_info = json.loads(line)
            image_path_A = image_info["image_path_A"]
            image_path_B = image_info["image_path_B"]

            candidates = []
            for llm in llms:
                try:
                    cap = generate_caption_from_images(image_path_A, image_path_B, llm, args)
                except Exception as e:
                    cap = f"Error: {e}"
                candidates.append(cap)
                time.sleep(args.delay)

            options = dict(zip(labels, candidates))

            try:
                best_label = judge_best_option(options, args, image_path_A, image_path_B)
            except Exception as e:
                best_label = None
                print("Judge error:", e, file=sys.stderr)
            time.sleep(args.delay)

            try:
                scores = score_options(options, args, image_path_A, image_path_B)
            except Exception as e:
                scores = {lbl: None for lbl in labels}
                print("Scoring error:", e, file=sys.stderr)
            time.sleep(args.delay)

            record = {
                "Image A": image_path_A,
                "Image B": image_path_B,
                "Options": options,
                "Judge Choice": best_label,
                "Scores": scores
            }
            fout_all.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout_all.flush()

            best_record = {
                "Image A": image_path_A,
                "Image B": image_path_B,
                "Best Caption": options.get(best_label, "")
            }
            fout_best.write(json.dumps(best_record, ensure_ascii=False) + "\n")
            fout_best.flush()

    fout_all.close()
    fout_best.close()


if __name__ == "__main__":
    main()
