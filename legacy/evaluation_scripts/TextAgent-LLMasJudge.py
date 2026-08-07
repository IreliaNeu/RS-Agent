#!/usr/bin/env python3
"""
Automate LLM-as-Judge evaluation for remote sensing change detection captions.

Workflow per original caption:
1. Call multiple OpenRouter LLMs to get refined descriptions.
2. Label outputs as A, B, C… and save a mapping file.
3. Use GPT-4o as Judge: pick the best option (only return the letter).
4. Use GPT-4o to score each option (1–10) and return a JSON score table.
5. Save the best caption per original to a JSONL file.
6. All hyper-parameters and file paths are configurable via args.
"""

import argparse
import json
import time
import requests
import random
# import openai
import sys
import httpx
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge pipeline for diversifying remote sensing captions"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input JSONL file (each line: {'Original Caption': ...})")
    parser.add_argument("--llm_config", "-c", required=True,
                        help="Path to JSON config listing OpenRouter LLM endpoints and keys")
    parser.add_argument("--mapping_output", "-m", required=True,
                        help="Path to output JSON file mapping labels to LLM names")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output JSONL file with full evaluation records")
    parser.add_argument("--best_output", "-b", required=True,
                        help="Path to JSONL file with only the best caption per original")
    # GPT-4o Judge settings
    parser.add_argument("--judge_model", 
                            # default="openai/gpt-4o-2024-11-20"
                            default="gpt-4o"
                            )
    parser.add_argument("--judge_api_key", required=True,
                        help="API key for OpenAI (Judge and scoring)")
    parser.add_argument("--judge_temperature", type=float, default=0.0,
                        help="Temperature for Judge calls")
    parser.add_argument("--judge_max_tokens", type=int, default=5,
                        help="Max tokens for Judge (should be very small)")
    # scoring settings
    parser.add_argument("--score_temperature", type=float, default=0.7,
                        help="Temperature for scoring calls")
    parser.add_argument("--score_max_tokens", type=int, default=60,
                        help="Max tokens for scoring output JSON")
    # pacing
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between API calls")
    return parser.parse_args()


def load_llm_configs(path):
    """Load list of LLM configs from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["llms"]

def call_openrouter(llm, messages, temperature, max_tokens, max_retries=3, retry_delay=2.0):
    """
    Enhanced version with retry mechanism and error handling
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
            print(f"[⚠️ Attempt {attempt}/{max_retries}] Error calling {llm['name']}: {str(e)}", 
                  file=sys.stderr)
            if attempt < max_retries:
                sleep_time = retry_delay * (2 ** (attempt-1)) + random.uniform(0, 0.5)
                print(f"Retrying in {sleep_time:.1f} seconds...", file=sys.stderr)
                time.sleep(sleep_time)
    
    error_msg = f"🚨 Failed after {max_retries} attempts for {llm['name']}"
    print(error_msg, file=sys.stderr)
    return f"Error: {error_msg}"

#引入CoT:1.Refine and enrich the description,2.Return only the refined caption
# system_prompt = (
#     "You are a professional remote sensing change detection caption editor. "
#     "Given a short and possibly incomplete or rough caption describing image change, "
#     "first analyze how the original caption could be improved in terms of clarity, completeness, or conciseness. "
#     "Then, refine and enrich the caption without changing its core meaning. "
#     "Return your reasoning as a short paragraph, followed by the final refined caption."
# )
# system_prompt = (
#     "You are a professional remote sensing change detection caption editor. "
#     "Your task is to refine and enrich short, rough captions describing land surface changes detected in remote sensing imagery. "
#     "You should first analyze the original caption to identify potential improvements in clarity, completeness, or scientific terminology. "
#     "Then, using your background knowledge in remote sensing and typical land use/cover change patterns, rewrite the caption to be more precise and informative without changing its original meaning. "
#     "When appropriate, incorporate domain knowledge (e.g., urban expansion, deforestation, seasonal flooding) to improve clarity. "
#     "Respond in JSON format with two fields: 'reasoning' and 'refined_caption'."
# )
def refine_caption(original, llm, args):
    """Generate one refined caption from a given LLM."""
    system_prompt = (
        "You are a professional remote sensing change detection caption editor. "
        "Refine and enrich the description without altering its meaning. "
        "Return only the refined caption text."
    )
    user_prompt = f"""
            Original Description:
            \"\"\"{original}\"\"\"
            Please provide one enriched but faithful rephrasing.
            """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()}
    ]
    return call_openrouter(
        llm, messages,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens * 4  # allow enough space
    )


# def judge_best_option(options, args):
#     """Ask GPT-4o to pick the best option letter."""
#     openai.api_key = args.judge_api_key
#     system = (
#         "You are an expert evaluator of remote sensing change detection descriptions."
#     )
#     # Build the multiple choice prompt
#     choices_text = "\n".join(f"{k}. {v}" for k, v in options.items())
#     user = (
#         "We have multiple refined captions for the same remote sensing change detection:"
#         f"\n\n{choices_text}\n\n"
#         "Which one provides the most information while preserving original meaning? "
#         "Return only the single letter (A, B, C, etc)."
#     )
#     resp = openai.ChatCompletion.create(
#         model=args.judge_model,
#         messages=[{"role": "system", "content": system},
#                   {"role": "user", "content": user}],
#         temperature=args.judge_temperature,
#         max_tokens=args.judge_max_tokens,
#         n=1,
#     )
#     return resp.choices[0].message.content.strip().upper()


# def score_options(options, args):
#     """Ask GPT-4o to score each option 1–10; return mapping letter→int."""
#     openai.api_key = args.judge_api_key
#     system = (
#         "You are an expert evaluator. Rate each caption 1 (worst) to 10 (best) "
#         "for richness and fidelity to original."
#     )
#     choices_text = "\n".join(f"{k}. {v}" for k, v in options.items())
#     user = (
#         "Here are the options:\n\n"
#         f"{choices_text}\n\n"
#         "Please return a JSON object mapping each letter to an integer score, "
#         "e.g. {\"A\":8, \"B\":7, \"C\":9}."
#     )
#     resp = openai.ChatCompletion.create(
#         model=args.judge_model,
#         messages=[{"role": "system", "content": system},
#                   {"role": "user", "content": user}],
#         temperature=args.score_temperature,
#         max_tokens=args.score_max_tokens,
#         n=1,
#     )
#     content = resp.choices[0].message.content.strip()
#     try:
#         return json.loads(content)
#     except json.JSONDecodeError:
#         print("⚠️ Failed to parse scoring JSON:", content, file=sys.stderr)
#         return {k: None for k in options}

def judge_best_option(options, args, max_retries=3):
    """Ask GPT-4o to pick the best option letter with retry logic."""
    # url = "https://openrouter.ai/api/v1/chat/completions"
    url = "http://chatapi.littlewheat.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.judge_api_key}",
        "Content-Type": "application/json"
    }

    system = "You are an expert evaluator of remote sensing change detection descriptions."
    choices_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = (
        "We have multiple refined captions for the same remote sensing change detection:"
        f"\n\n{choices_text}\n\n"
        "Which one provides the most information while preserving original meaning? "
        "Return only the single letter (A, B, C, etc)."
    )

    payload = {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": args.judge_temperature,
        "max_tokens": args.judge_max_tokens,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content.strip().upper()
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = args.delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"⚠️ [Judge] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {delay:.1f}s...",
                      file=sys.stderr)
                time.sleep(delay)
    
    error_msg = f"🚨 Judge failed after {max_retries} attempts: {str(last_error)}"
    print(error_msg, file=sys.stderr)
    return None

import re

def score_options(options, args, max_retries=3):
    """Ask GPT-4o to score each option 1-10 with retry logic."""
    url = "http://chatapi.littlewheat.com/v1/chat/completions"
    # url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.judge_api_key}",
        "Content-Type": "application/json"
    }

    system = (
        "You are an expert evaluator. Rate each caption 1 (worst) to 10 (best) "
        "for richness and fidelity to original."
    )
    choices_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = (
        "Here are the options:\n\n"
        f"{choices_text}\n\n"
        "Please return a JSON object mapping each letter to an integer score, "
        "e.g. {{\"A\":8, \"B\":7, \"C\":9}}."
    )

    payload = {
        "model": args.judge_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": args.score_temperature,
        "max_tokens": args.score_max_tokens,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    last_error = f"JSON parse error: {str(e)}"
            else:
                last_error = f"No JSON found in response: {content[:100]}..."
            
            if attempt < max_retries - 1:
                delay = args.delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"⚠️ [Scoring] Attempt {attempt + 1}/{max_retries} failed: {last_error}. Retrying in {delay:.1f}s...",
                      file=sys.stderr)
                time.sleep(delay)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = args.delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"⚠️ [Scoring] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {delay:.1f}s...",
                      file=sys.stderr)
                time.sleep(delay)
    
    error_msg = f"🚨 Scoring failed after {max_retries} attempts: {str(last_error)}"
    print(error_msg, file=sys.stderr)
    return {k: None for k in options}

def main():
    args = parse_args()

    # Load configs and labels
    llms = load_llm_configs(args.llm_config)
    labels = [chr(ord("A") + i) for i in range(len(llms))]
    mapping = {label: llm["name"] for label, llm in zip(labels, llms)}

    # Save the mapping file once
    with open(args.mapping_output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Prepare file handles
    fout_all = open(args.output, "w", encoding="utf-8")
    fout_best = open(args.best_output, "w", encoding="utf-8")

    # Process each original caption
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            orig = json.loads(line).get("Original Caption", "").strip()
            # 1. Generate candidates
            candidates = []
            for llm in llms:
                try:
                    cand = refine_caption(orig, llm, args)
                except Exception as e:
                    cand = f"Error: {e}"
                candidates.append(cand)
                time.sleep(args.delay)

            options = dict(zip(labels, candidates))

            # 2. Judge best
            try:
                best_label = judge_best_option(options, args)
            except Exception as e:
                best_label = None
                print("Judge error:", e, file=sys.stderr)

            time.sleep(args.delay)

            # 3. Score all
            try:
                scores = score_options(options, args)
            except Exception as e:
                scores = {lbl: None for lbl in labels}
                print("Scoring error:", e, file=sys.stderr)

            time.sleep(args.delay)

            # 4. Prepare records
            best_caption = options.get(best_label, None)
            record = {
                "Original Caption": orig,
                "Options": options,
                "Judge Choice": best_label,
                "Scores": scores
            }
            fout_all.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout_all.flush()

            best_record = {
                "Original Caption": orig,
                "Best Caption": best_caption
            }
            fout_best.write(json.dumps(best_record, ensure_ascii=False) + "\n")
            fout_best.flush()

    fout_all.close()
    fout_best.close()


if __name__ == "__main__":
    main()