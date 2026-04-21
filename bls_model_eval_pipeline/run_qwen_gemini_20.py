#!/usr/bin/env python3
"""
Run corrected A/B prompt evaluation across 20 representative BLS questions.

Models (default):
- Qwen/Qwen2.5-VL-72B-Instruct (NCSA-hosted via chat.illinois.edu)
- gemini-3-flash-preview
- gemini-3.1-flash-lite-preview

Outputs:
- bls_model_eval_pipeline/results/prompt_v4_runs_20.json
- bls_model_eval_pipeline/results/prompt_v4_compliance_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PIPELINE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Import runtime prompt + model dispatch from the app so evaluation matches app behavior.
import sys

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import ADMISSIONS_FAQ_URL, PROGRAM_URL, SYSTEM_PROMPT_A_V3, SYSTEM_PROMPT_B_V3, call_model  # noqa: E402
from bls_model_eval_pipeline.questions import get_all_questions_flat  # noqa: E402


# 20-question representative set, sourced from questions.py by stable key.
QUESTION_KEYS_20 = [
    "Prospective Student Questions|General|1",
    "Prospective Student Questions|General|2",
    "Prospective Student Questions|General|4",
    "Prospective Student Questions|Admissions|5",
    "Prospective Student Questions|Admissions|6",
    "Prospective Student Questions|Admissions|8",
    "Prospective Student Questions|Admissions|9",
    "Prospective Student Questions|Academics|11",
    "Prospective Student Questions|Academics|14",
    "Prospective Student Questions|Academics|13",
    "Prospective Student Questions|Cost & Financial Aid|18",
    "Prospective Student Questions|Cost & Financial Aid|20",
    "Prospective Student Questions|Cost & Financial Aid|22",
    "Current Student Questions|Advising & Registration|1",
    "Current Student Questions|Policies|8",
    "More Subjective/Open-Ended Questions||1",
    "More Subjective/Open-Ended Questions||3",
    "More Subjective/Open-Ended Questions||4",
    "More Subjective/Open-Ended Questions||13",
    "More Subjective/Open-Ended Questions||15",
]


CHECK_RULES = [
    ("self_paced_language", re.compile(r"\bself[- ]?paced\b", re.IGNORECASE)),
    ("adult_learner_language", re.compile(r"\badult learner(s)?\b", re.IGNORECASE)),
    ("non_traditional_language", re.compile(r"\bnon[- ]traditional\b", re.IGNORECASE)),
    ("future_launch_reference", re.compile(r"\b(will|scheduled to)\s+launch\b|\bfall\s*2026\b", re.IGNORECASE)),
    ("transfer_credit_friendly_phrase", re.compile(r"transfer[- ]credit friendly", re.IGNORECASE)),
    ("max_time_completion", re.compile(r"\b(maximum|up to)\b[^.\n]{0,50}\b(8 years|time to completion)\b", re.IGNORECASE)),
    ("billing_hour_term", re.compile(r"\bbilling hour(s)?\b", re.IGNORECASE)),
    ("escaped_dollar_symbol", re.compile(r"\\\$")),
    ("demographic_targeting", re.compile(r"\b(black|latinx|ages?\s*\d+\s*[-–]\s*\d+|underrepresented|marginalized)\b", re.IGNORECASE)),
    ("text_message_advising", re.compile(r"\btext (message|messaging)\b", re.IGNORECASE)),
    ("internal_acronym_exposed", re.compile(r"\bICT\b", re.IGNORECASE)),
    ("equivalence_claim", re.compile(r"\b(equally legitimate|equivalent in rigor|same rigor|just as rigorous)\b", re.IGNORECASE)),
]


def shorten(text: str, limit: int = 240) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def build_question_bank() -> list[dict[str, Any]]:
    by_key = {q["key"]: q for q in get_all_questions_flat()}
    missing = [k for k in QUESTION_KEYS_20 if k not in by_key]
    if missing:
        raise RuntimeError(f"Missing keys in questions.py: {missing}")
    return [by_key[k] for k in QUESTION_KEYS_20]


def evaluate_answer(answer: str, question_key: str) -> list[str]:
    violations: list[str] = []
    text = answer or ""
    lower = text.lower()

    for name, pattern in CHECK_RULES:
        if pattern.search(text):
            violations.append(name)

    # Dynamic-detail checks for specific question types.
    if question_key.endswith("|9") and "Admissions|9" in question_key:
        # Application deadlines question should avoid hardcoded dates and include dynamic source direction.
        if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", lower):
            violations.append("hardcoded_deadline_months")
        if ADMISSIONS_FAQ_URL.lower() not in lower and "admissions" not in lower:
            violations.append("missing_admissions_faq_direction")

    if "Academics|11" in question_key:
        # Concentrations question should avoid stale hardcoded lists without dynamic pointer.
        mentions_known_list = all(term in lower for term in ["global", "health", "management"])
        if mentions_known_list and PROGRAM_URL.lower() not in lower:
            violations.append("possible_hardcoded_concentration_list")

    if "Advising & Registration|1" in question_key:
        if "office of undergraduate admissions" not in lower:
            violations.append("missing_office_of_undergraduate_admissions")

    return sorted(set(violations))


def is_error_message(text: str) -> bool:
    return bool(re.match(r"^(API_|COURSE_NOT_FOUND|MISSING_GEMINI_API_KEY|Error:)", (text or "").strip()))


def run_for_model(model: str, questions: list[dict[str, Any]], sleep_seconds: float = 0.4) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, q in enumerate(questions, start=1):
        question_text = q["text"]
        question_key = q["key"]
        print(f"[{model}] [{idx}/{len(questions)}] {question_text}")

        t0 = time.time()
        single_answer = call_model(SYSTEM_PROMPT_B_V3, question_text, model=model)
        single_latency = round(time.time() - t0, 2)

        t1 = time.time()
        draft_answer = call_model(SYSTEM_PROMPT_A_V3, question_text, model=model)
        refine_input = (
            f"Original User Query: {question_text}\n"
            f"Internal Draft Answer: {draft_answer}\n\n"
            "Please refine this draft into a final response following your system constraints."
        )
        dual_answer = call_model(SYSTEM_PROMPT_B_V3, refine_input, model=model)
        dual_latency = round(time.time() - t1, 2)

        row = {
            "question_key": question_key,
            "question": question_text,
            "single": {
                "answer": single_answer,
                "latency": single_latency,
                "ok": not is_error_message(single_answer),
                "violations": evaluate_answer(single_answer, question_key),
            },
            "dual": {
                "draft": draft_answer,
                "answer": dual_answer,
                "latency": dual_latency,
                "ok": (not is_error_message(draft_answer)) and (not is_error_message(dual_answer)),
                "violations": evaluate_answer(dual_answer, question_key),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rows.append(row)
        time.sleep(sleep_seconds)

    return {"model": model, "rows": rows}


def summarize_compliance(model_runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model_run in model_runs:
        model = model_run["model"]
        rows = model_run["rows"]
        dual_violations = {}
        single_violations = {}
        dual_ok = 0
        single_ok = 0

        for row in rows:
            if row["single"]["ok"]:
                single_ok += 1
            if row["dual"]["ok"]:
                dual_ok += 1

            for v in row["single"]["violations"]:
                single_violations[v] = single_violations.get(v, 0) + 1
            for v in row["dual"]["violations"]:
                dual_violations[v] = dual_violations.get(v, 0) + 1

        summary[model] = {
            "questions": len(rows),
            "single_ok_count": single_ok,
            "dual_ok_count": dual_ok,
            "single_violations": dict(sorted(single_violations.items(), key=lambda kv: kv[1], reverse=True)),
            "dual_violations": dict(sorted(dual_violations.items(), key=lambda kv: kv[1], reverse=True)),
        }
    return summary


def write_markdown_report(path: Path, model_runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Prompt v4 Compliance Report (20 Questions)")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Admissions FAQ: {ADMISSIONS_FAQ_URL}")
    lines.append(f"- Program page: {PROGRAM_URL}")
    lines.append("")

    for model_run in model_runs:
        model = model_run["model"]
        lines.append(f"## Model: `{model}`")
        model_summary = summary.get(model, {})
        lines.append("")
        lines.append(f"- Questions: {model_summary.get('questions', 0)}")
        lines.append(f"- Single-agent successful calls: {model_summary.get('single_ok_count', 0)}")
        lines.append(f"- Dual-agent successful calls: {model_summary.get('dual_ok_count', 0)}")
        lines.append(f"- Single-agent violation counts: {json.dumps(model_summary.get('single_violations', {}), ensure_ascii=False)}")
        lines.append(f"- Dual-agent violation counts: {json.dumps(model_summary.get('dual_violations', {}), ensure_ascii=False)}")
        lines.append("")
        lines.append("| # | Question | Dual-Agent Snippet | Dual Violations |")
        lines.append("| ---: | --- | --- | --- |")

        for i, row in enumerate(model_run["rows"], start=1):
            q = row["question"].replace("|", "\\|")
            snippet = shorten(row["dual"]["answer"]).replace("|", "\\|")
            viol = ", ".join(row["dual"]["violations"]) or "none"
            lines.append(f"| {i} | {q} | {snippet} | {viol} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="Qwen/Qwen2.5-VL-72B-Instruct,gemini-3-flash-preview,gemini-3.1-flash-lite-preview",
        help="Comma-separated model list",
    )
    parser.add_argument(
        "--output-json",
        default=str(RESULTS_DIR / "prompt_v4_runs_20.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        default=str(RESULTS_DIR / "prompt_v4_compliance_report.md"),
        help="Output markdown report path",
    )
    parser.add_argument("--sleep", type=float, default=0.4, help="Delay between question runs")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    questions = build_question_bank()

    all_runs = []
    for model in models:
        all_runs.append(run_for_model(model=model, questions=questions, sleep_seconds=args.sleep))

    summary = summarize_compliance(all_runs)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "question_keys": QUESTION_KEYS_20,
            "admissions_faq_url": ADMISSIONS_FAQ_URL,
            "program_url": PROGRAM_URL,
        },
        "summary": summary,
        "runs": all_runs,
    }
    out_json.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(out_md, all_runs, summary)

    print(f"Wrote JSON: {out_json}")
    print(f"Wrote Markdown: {out_md}")


if __name__ == "__main__":
    main()
