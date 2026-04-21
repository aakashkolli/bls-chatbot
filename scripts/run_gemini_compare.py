#!/usr/bin/env python3
"""
Run Gemini model(s) and compare with NCSA-hosted open-source model endpoints
over the question bank in `bls_model_eval_pipeline/questions.py`.

Usage examples:
  export GEMINI_API_KEY=YOUR_KEY
  python scripts/run_gemini_compare.py --models gemini-3-flash-preview,gemini-3.1-flash-lite-preview \
      --ncsa ncsa_local=http://localhost:8080/generate

The script reads system prompts from `config/system_prompts.json` (created with
defaults on first run). Edit that file to change system prompts for each model.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests


DEFAULT_CONFIG_PATH = os.path.join("config", "system_prompts.json")


def load_system_prompts(path=DEFAULT_CONFIG_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # create defaults
    defaults = {
        "gemini-3-flash-preview": "You are an assistant specialized in answering prospective student questions about the Bachelor of Liberal Studies (BLS). Answer concisely, accurately, and in a student-facing tone. If the question asks for procedural steps, include a clear next step.",
        "gemini-3.1-flash-lite-preview": "You are an assistant specialized in answering prospective student questions about the Bachelor of Liberal Studies (BLS). Answer concisely, accurately, and in a student-facing tone. If the question asks for procedural steps, include a clear next step.",
        "ncsa_default": "You are an assistant answering BLS program FAQ-style questions. Be concise and helpful."
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2)
    print(f"Wrote default system prompts to {path}. Edit this file to customize prompts.")
    return defaults


def get_questions(use_subset=False):
    # import the questions module from the pipeline
    try:
        from bls_model_eval_pipeline import questions as qmod
    except Exception:
        sys.path.insert(0, os.getcwd())
        from bls_model_eval_pipeline import questions as qmod

    if use_subset and hasattr(qmod, "get_subset_questions"):
        return qmod.get_subset_questions()
    return qmod.get_all_questions_flat()


def extract_text_from_response(obj):
    # Try to find a useful text field inside a JSON response.
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # common top-level keys
        if "candidates" in obj and isinstance(obj["candidates"], list) and obj["candidates"]:
            c = obj["candidates"][0]
            if isinstance(c, dict):
                # candidate may have 'content' list with parts
                content = c.get("content") or c.get("output") or c.get("text")
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)
                    if texts:
                        return "\n".join(texts)
                if isinstance(content, str):
                    return content
        # try other common fields
        for key in ("output", "outputs", "choices", "response", "responses", "results", "data"):
            if key in obj:
                v = obj[key]
                if isinstance(v, list) and v:
                    first = v[0]
                    if isinstance(first, dict):
                        for tk in ("text", "generated_text", "content", "answer"):
                            if tk in first and isinstance(first[tk], str):
                                return first[tk]
                    elif isinstance(first, str):
                        return first
                elif isinstance(v, str):
                    return v

    # recursive search for first string-like value for likely keys
    def find(obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("text", "generated_text", "content", "answer", "response") and isinstance(v, str):
                    return v
                found = find(v)
                if found:
                    return found
        if isinstance(obj, list):
            for it in obj:
                found = find(it)
                if found:
                    return found
        return None

    found = find(obj)
    if found:
        return found
    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)


def call_gemini(model, api_key, prompt, timeout=30):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-goog-api-key"] = api_key
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}
    try:
        j = r.json()
    except Exception:
        return {"http_status": r.status_code, "text": r.text}
    return j


def call_ncsa(endpoint, prompt, headers=None, timeout=30):
    # Generic POST with JSON body using key 'prompt' (many open-source hosts accept this; adapt as needed)
    payload = {"prompt": prompt}
    try:
        r = requests.post(endpoint, json=payload, headers=(headers or {}), timeout=timeout)
    except Exception as e:
        return {"error": str(e)}
    try:
        return r.json()
    except Exception:
        return {"http_status": r.status_code, "text": r.text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--models", default="gemini-3-flash-preview,gemini-3.1-flash-lite-preview", help="Comma-separated Gemini model IDs")
    parser.add_argument("--ncsa", default="", help="Comma-separated NCSA endpoints in name=url form (e.g. local=http://localhost:8080/generate)")
    parser.add_argument("--ncsa-headers", default="", help="Optional JSON string of headers to add to NCSA requests")
    parser.add_argument("--use-subset", action="store_true", help="Use representative 12-question subset")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to system prompts JSON config")
    parser.add_argument("--output", default="results/gemini_ncsa_comparison.json", help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: No Gemini API key provided. Set --api-key or export GEMINI_API_KEY.")

    system_prompts = load_system_prompts(args.config)

    questions = get_questions(use_subset=args.use_subset)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    ncsa_map = {}
    if args.ncsa:
        for pair in args.ncsa.split(","):
            if "=" in pair:
                name, url = pair.split("=", 1)
                ncsa_map[name.strip()] = url.strip()
            else:
                print(f"Skipping invalid ncsa entry: {pair}")

    ncsa_headers = {}
    if args.ncsa_headers:
        try:
            ncsa_headers = json.loads(args.ncsa_headers)
        except Exception:
            print("Failed to parse --ncsa-headers JSON; ignoring headers")

    results = {"meta": {"timestamp": datetime.utcnow().isoformat() + "Z", "models": models, "ncsa": list(ncsa_map.keys())}, "questions": []}

    for q in questions:
        qobj = {"section": q.get("section"), "subsection": q.get("subsection"), "num": q.get("num"), "text": q.get("text"), "responses": []}
        for model in models:
            sp = system_prompts.get(model) or system_prompts.get("defaults") or ""
            prompt = f"{sp}\n\nQuestion: {q.get('text')}\nAnswer:"
            raw = call_gemini(model, api_key, prompt)
            text = extract_text_from_response(raw)
            qobj["responses"].append({"model": model, "text": text, "raw": raw})

        for name, endpoint in ncsa_map.items():
            sp = system_prompts.get(name) or system_prompts.get("ncsa_default") or ""
            prompt = f"{sp}\n\nQuestion: {q.get('text')}\nAnswer:"
            raw = call_ncsa(endpoint, prompt, headers=ncsa_headers)
            text = extract_text_from_response(raw)
            qobj["responses"].append({"model": name, "text": text, "raw": raw})

        results["questions"].append(qobj)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote comparison results to {args.output}")


if __name__ == "__main__":
    main()
