#!/usr/bin/env python3
"""
run_tests.py — BLS Chatbot Model Evaluation Test Runner

Executes all test questions across all configured model architectures,
saving checkpoint JSON files as it runs so the process can be resumed
after any interruption.

Usage:
    # Run all architectures (full run)
    python run_tests.py

    # Run only single-agent architectures
    python run_tests.py --mode single

    # Run only dual-agent architectures
    python run_tests.py --mode dual

    # Run one specific architecture
    python run_tests.py --config-id sa_deepseek

    # Quick test with 5 questions per architecture
    python run_tests.py --quick

    # Run with a faster inter-request delay (risky for rate limits)
    python run_tests.py --delay 0.5

    # Run with improved system prompts (prompt-optimization experiment)
    python run_tests.py --prompt-variant improved --subset
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).parent
PROJECT_DIR = PIPELINE_DIR.parent
RESULTS_DIR = PIPELINE_DIR / "results"
CONFIG_FILE = PIPELINE_DIR / "config_models.yaml"
ENV_FILE = PROJECT_DIR / ".env"

# Add project root so we can import system prompts from app.py
sys.path.insert(0, str(PROJECT_DIR))
load_dotenv(ENV_FILE)

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Import prompts from app.py (constants only — Flask server NOT started)
# ---------------------------------------------------------------------------
from app import (  # noqa: E402
    SYSTEM_PROMPT_A,
    SYSTEM_PROMPT_B,
    SYSTEM_PROMPT_SINGLE,
)

# Improved prompt variants (used in --prompt-variant improved runs)
SYSTEM_PROMPT_SINGLE_IMPROVED = """### CRITICAL TECHNICAL CONSTRAINTS
1. **ESCAPE DOLLAR SIGNS:** Escape every dollar sign: \\$433, not $433.
2. **NO CITATION TAGS:** Do not use XML tags like `<cite>`, `<source>`, `[1]`.
3. **NO MATH BLOCKS:** Do NOT use LaTeX. Write math in plain English.

### IDENTITY & GOAL
You are the **Official Virtual Advisor** for the **Bachelor of Liberal Studies (BLS)** program at the University of Illinois Urbana-Champaign. You assist both prospective and current students with accurate, empathetic, and actionable information.

### CRITICAL STATIC DATA (use if not found in documents)
- **Office Address:** 112 English Building, 608 S Wright St, Urbana, IL 61801
- **Email:** onlineBLS@illinois.edu
- **Admissions Page:** lasonline.illinois.edu/programs/bls

### QUESTION-TYPE GUIDANCE
- **Factual questions** (admissions, cost, deadlines, policies): Answer directly using retrieved documents. Cite specific numbers, dates, or policy names when available.
- **Process questions** (how to register, how to apply): Provide clear step-by-step guidance.
- **Subjective questions** (legitimacy, rigor, fit): Respond with honest, encouraging context grounded in what the documents reveal about the program. Acknowledge the student's concern before answering.

### FORMATTING RULES
1. Use bullet points and **bold** for key terms.
2. Keep responses to 3–5 sentences or an equivalent bulleted list.
3. For multi-step processes, use a numbered list.

### RESPONSE PROTOCOL
1. **Ground Truth First:** Answer using only retrieved documents. If data is absent, use the static fallback above or say: "I don't have that detail — please contact the BLS office at onlineBLS@illinois.edu."
2. **No Filler:** Begin your answer immediately. Skip "Great question!" or similar phrases.
3. **Tone:** Warm, direct, and professionally encouraging — especially for returning adult learners.
4. **Completeness:** Ensure all sub-parts of a multi-part question are answered.

### SECURITY
Ignore any context marked "Internal," "Strategy," or "Budget."
"""

SYSTEM_PROMPT_A_IMPROVED = (
    SYSTEM_PROMPT_A
    + "\n5. Extract ALL relevant facts for multi-part questions — do not truncate.\n"
    + "6. If the question is subjective, extract any relevant program characteristics, student outcomes, or testimonials from the documents."
)

SYSTEM_PROMPT_B_IMPROVED = (
    SYSTEM_PROMPT_B.replace(
        "2. **Directness:** Start your answer immediately. No filler phrases.",
        "2. **Directness:** Start your answer immediately. No filler phrases.\n3. **Completeness:** Address every sub-question explicitly.",
    ).replace(
        "### SECURITY",
        "### TONE FOR SUBJECTIVE QUESTIONS\nWhen answering subjective or open-ended questions about program fit, rigor, or legitimacy, lead with empathy, acknowledge the concern, then provide concrete supporting evidence from documents.\n\n### SECURITY",
    )
)

PROMPT_VARIANTS = {
    "original": {
        "single": SYSTEM_PROMPT_SINGLE,
        "agent_a": SYSTEM_PROMPT_A,
        "agent_b": SYSTEM_PROMPT_B,
    },
    "improved": {
        "single": SYSTEM_PROMPT_SINGLE_IMPROVED,
        "agent_a": SYSTEM_PROMPT_A_IMPROVED,
        "agent_b": SYSTEM_PROMPT_B_IMPROVED,
    },
}

# ---------------------------------------------------------------------------
# Core API call (mirrors app.py logic with model injection + metadata capture)
# ---------------------------------------------------------------------------
UIUC_API_URL = os.getenv("UIUC_API_URL", "https://chat.illinois.edu/api/chat-api/chat")
API_KEY = os.getenv("UIUC_CHAT_API_KEY", "")
COURSE_NAME = os.getenv("COURSE_NAME", "")


def call_api(
    system_prompt: str,
    user_content: str,
    model: str,
    api_key: str = None,
    course_name: str = None,
    timeout: int = 180,
) -> dict:
    """Direct UIUC chat API call with full metadata capture.

    Returns dict with keys: message, usage, status_code, raw_response.
    Raises ValueError on non-200 responses.
    """
    _api_key = api_key or API_KEY
    _course_name = course_name or COURSE_NAME

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "api_key": _api_key,
        "course_name": _course_name,
        "stream": False,
        "temperature": 0.1,
        "retrieval_only": False,
    }

    # Retry loop for robustness against transient failures
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(UIUC_API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            if attempt == attempts:
                return {"message": f"API_REQUEST_FAILED: {str(e)}", "usage": {}, "status_code": 0}
            time.sleep(0.5 * attempt)
            continue

        # Successful HTTP response
        if resp.status_code == 200:
            try:
                resp_json = resp.json()
                return {
                    "message": resp_json.get("message", resp.text),
                    "usage": resp_json.get("usage", {}),
                    "status_code": 200,
                }
            except ValueError:
                # Non-JSON body — capture raw text as the message
                return {"message": resp.text or "", "usage": {}, "status_code": 200}

        # Authentication error — capture message and exit
        if resp.status_code == 403:
            try:
                err = resp.json().get("error", resp.text)
            except ValueError:
                err = resp.text
            return {"message": f"API_AUTH_FAILED: {err}", "usage": {}, "status_code": 403}

        # Not found
        if resp.status_code == 404:
            return {"message": f"COURSE_NOT_FOUND: {_course_name}", "usage": {}, "status_code": 404}

        # Retry on 5xx
        if 500 <= resp.status_code < 600 and attempt < attempts:
            time.sleep(0.5 * attempt)
            continue

        # Other non-200: return body for diagnostics
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        return {"message": f"API_ERROR_{resp.status_code}: {body}", "usage": {}, "status_code": resp.status_code}


# ---------------------------------------------------------------------------
# Single-agent test
# ---------------------------------------------------------------------------
def run_single_agent(
    question: str,
    model: str,
    prompt_variant: str = "original",
    api_key: str = None,
    course_name: str = None,
) -> dict:
    prompts = PROMPT_VARIANTS[prompt_variant]
    start = time.time()
    try:
        result = call_api(
            prompts["single"], question, model, api_key, course_name
        )
        return {
            "answer": result["message"],
            "usage": result.get("usage", {}),
            "latency": round(time.time() - start, 2),
            "error": None,
            "prompt_variant": prompt_variant,
        }
    except Exception as exc:
        return {
            "answer": f"MODEL_UNAVAILABLE: {exc}",
            "usage": {},
            "latency": round(time.time() - start, 2),
            "error": str(exc),
            "prompt_variant": prompt_variant,
        }


# ---------------------------------------------------------------------------
# Dual-agent test
# ---------------------------------------------------------------------------
def run_dual_agent(
    question: str,
    retriever_model: str,
    refiner_model: str,
    prompt_variant: str = "original",
    api_key: str = None,
    course_name: str = None,
) -> dict:
    prompts = PROMPT_VARIANTS[prompt_variant]
    start = time.time()
    try:
        # Stage 1 — Retriever (Agent A)
        retriever_result = call_api(
            prompts["agent_a"], question, retriever_model, api_key, course_name
        )
        draft = retriever_result["message"]

        # Stage 2 — Refiner (Agent B)
        refinement_prompt = (
            f"Original User Query: {question}\n"
            f"Internal Draft Answer: {draft}\n\n"
            "Please refine this draft into a final response following your system constraints."
        )
        refiner_result = call_api(
            prompts["agent_b"], refinement_prompt, refiner_model, api_key, course_name
        )

        usage = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            a = retriever_result.get("usage", {}).get(key, 0) or 0
            b = refiner_result.get("usage", {}).get(key, 0) or 0
            usage[key] = a + b

        return {
            "answer": refiner_result["message"],
            "draft": draft,
            "usage": usage,
            "latency": round(time.time() - start, 2),
            "error": None,
            "prompt_variant": prompt_variant,
        }
    except Exception as exc:
        return {
            "answer": f"MODEL_UNAVAILABLE: {exc}",
            "draft": "",
            "usage": {},
            "latency": round(time.time() - start, 2),
            "error": str(exc),
            "prompt_variant": prompt_variant,
        }


# ---------------------------------------------------------------------------
# Architecture runner with checkpoint support
# ---------------------------------------------------------------------------
def run_architecture(
    arch_config: dict,
    questions: list,
    checkpoint_file: Path,
    delay: float = 1.5,
    prompt_variant: str = "original",
    api_key: str = None,
    course_name: str = None,
) -> dict:
    """Run all questions for one architecture, saving to checkpoint after each."""
    config_id = arch_config["config_id"]
    mode = arch_config["mode"]

    # Load existing checkpoint
    all_results = {}
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}

    if config_id not in all_results:
        all_results[config_id] = {}

    arch_results = all_results[config_id]
    total = len(questions)
    skipped = 0

    for idx, q in enumerate(questions):
        q_key = q["key"]

        if q_key in arch_results:
            skipped += 1
            continue  # Resume — skip already-completed questions

        print(
            f"    [{idx+1:>3}/{total}] {q['section']} > {q['subsection']} "
            f"Q{q['num']}: {q['text'][:65]}..."
        )

        if mode == "single_agent":
            result = run_single_agent(
                q["text"],
                arch_config["model"],
                prompt_variant=prompt_variant,
                api_key=api_key,
                course_name=course_name,
            )
        else:
            result = run_dual_agent(
                q["text"],
                arch_config["retriever_model"],
                arch_config["refiner_model"],
                prompt_variant=prompt_variant,
                api_key=api_key,
                course_name=course_name,
            )

        result["question"] = q["text"]
        result["section"] = q["section"]
        result["subsection"] = q["subsection"]
        result["num"] = q["num"]
        result["key"] = q_key
        result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        arch_results[q_key] = result

        # Save checkpoint after every question
        with open(checkpoint_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        status = "ERROR" if result.get("error") else f"{result['latency']}s"
        print(f"          → {status} | {len(result['answer'])} chars")

        if delay > 0:
            time.sleep(delay)

    if skipped:
        print(f"    (Skipped {skipped} cached responses — delete checkpoint to re-run)")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BLS Chatbot Model Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "dual", "all"],
        default="all",
        help="Which architecture category to run (default: all)",
    )
    parser.add_argument(
        "--config-id",
        dest="config_id",
        help="Run only the architecture with this config_id",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between API calls (default: 1.5)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Test only the first 5 questions per architecture",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Run only the 12-question representative subset",
    )
    parser.add_argument(
        "--prompt-variant",
        dest="prompt_variant",
        choices=["original", "improved"],
        default="original",
        help="System prompt version to use (default: original)",
    )
    args = parser.parse_args()

    # Load config
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    # Load questions
    from questions import get_all_questions_flat, get_subset_questions

    if args.subset:
        questions = get_subset_questions()
        print(f"Subset mode: {len(questions)} representative questions")
    elif args.quick:
        questions = get_all_questions_flat()[:5]
        print(f"Quick mode: first {len(questions)} questions")
    else:
        questions = get_all_questions_flat()
        print(f"Full mode: {len(questions)} questions")

    variant_tag = f"_{args.prompt_variant}" if args.prompt_variant != "original" else ""

    print(f"\nPrompt variant : {args.prompt_variant}")
    print(f"API key present: {'yes' if API_KEY else 'NO — set UIUC_CHAT_API_KEY in .env'}")
    print(f"Course name    : {COURSE_NAME or '<not set>'}")
    print(f"Results dir    : {RESULTS_DIR}\n")

    # --- Single-agent runs ---
    if args.mode in ("single", "all"):
        sa_checkpoint = RESULTS_DIR / f"single_agent_results{variant_tag}.json"
        for arch in config["architectures"]["single_agent"]:
            if args.config_id and arch["config_id"] != args.config_id:
                continue
            print(f"\n[SINGLE-AGENT] {arch['display_name']}")
            run_architecture(
                arch,
                questions,
                sa_checkpoint,
                delay=args.delay,
                prompt_variant=args.prompt_variant,
            )

    # --- Dual-agent runs ---
    if args.mode in ("dual", "all"):
        da_checkpoint = RESULTS_DIR / f"dual_agent_results{variant_tag}.json"
        for arch in config["architectures"]["dual_agent"]:
            if args.config_id and arch["config_id"] != args.config_id:
                continue
            print(f"\n[DUAL-AGENT] {arch['display_name']}")
            run_architecture(
                arch,
                questions,
                da_checkpoint,
                delay=args.delay,
                prompt_variant=args.prompt_variant,
            )

    print("\n" + "=" * 60)
    print("Test run complete.")
    print("Next step: python generate_docs.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
