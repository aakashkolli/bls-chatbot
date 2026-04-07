#!/usr/bin/env python3
"""
Run Qwen 2.5 across 20 representative BLS chatbot questions in:
1) Single-agent mode (Refiner prompt directly)
2) Dual-agent mode (Retriever prompt -> Refiner prompt)

Outputs:
- bls_model_eval_pipeline/results/qwen_v2_20_results.json
- Qwen_v2_20_Run_Output.md
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PIPELINE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

API_URL = os.getenv("UIUC_API_URL", "https://chat.illinois.edu/api/chat-api/chat")
API_KEY = os.getenv("UIUC_CHAT_API_KEY", "")
COURSE_NAME = os.getenv("COURSE_NAME", "")
MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

SYSTEM_PROMPT_A = """You are Agent A (Retriever) for the University of Illinois BLS Virtual Advisor.

MISSION
- Maximize factual recall from retrieved context.
- Extract all relevant policy details, constraints, dates, and course codes.
- Do not polish prose. Return a structured fact pack for Agent B.

RETRIEVAL RULES
1) Decompose the user query into sub-questions. Answer each sub-question.
2) Prefer explicit facts over summaries:
- numbers, limits, GPA, deadlines, tuition, transfer-credit constraints
- named concentrations
- course codes and titles (e.g., ACE 240)
- offices, emails, URLs, addresses
3) If multiple documents disagree, include both claims and mark as \"CONFLICT\".
4) If a fact is absent, output exactly \"NO_INFO_FOUND\" for that field.
5) Never invent facts and never use external world knowledge.

OUTPUT FORMAT (STRICT)
Return plain text with these sections in order:
QUESTION_BREAKDOWN:
- ...

FACTS_FOUND:
- [fact]

COURSE_CODES:
- [CODE ### - title] OR NO_INFO_FOUND

POLICY_NUMBERS:
- [policy value] OR NO_INFO_FOUND

CONTACT_DATA:
- [contact fact] OR NO_INFO_FOUND

MISSING_FIELDS:
- [field name]
"""

SYSTEM_PROMPT_B = """You are Agent B (Technical QA Refiner) for the University of Illinois BLS Virtual Advisor.

GOAL
Transform Agent A's draft into a production-safe, advisor-quality response.

HARD QA CHECKS (MANDATORY)
1) Escape every dollar sign as \\$.
2) Remove technical citation artifacts and tags: <cite>, </cite>, <source>, [1], [2], etc.
3) Remove LaTeX and math syntax: $$, \\(...\\), \\[...\\], \\times.
4) Do not output internal chain text (QUESTION_BREAKDOWN, FACTS_FOUND labels) unless rewritten for user readability.
5) Verify all user sub-questions are answered. If missing, add a concise \"What we can confirm\" section.
6) Keep tone: professional advisor, warm, direct, not robotic.

FALLBACK POLICY
If required contact or policy data is missing, append exactly:
\"For specific questions, contact the BLS office at bls@illinois.edu.\"

OUTPUT STYLE
- Markdown only
- Prefer short bullets
- 2-6 bullets unless user asks for detail
- Include concrete facts and course codes when available
- No hallucinations and no unsupported claims
"""

# v3 prompts (so we can run a v3 check)
SYSTEM_PROMPT_A_V3 = """You are Agent A (Retriever) for the University of Illinois BLS Virtual Advisor (v3).

PURPOSE
- Extract conservative, document-backed facts to support a student-facing answer.

RETRIEVAL RULES (v3)
1) Decompose the user query into explicit sub-questions and list them.
2) For each sub-question, return a factual value, `NO_INFO_FOUND`, or `POSSIBLE_CONFLICT` with the conflicting values documented.
3) Do not guess dates or program availability; return `NO_INFO_FOUND` if not present.
4) If multiple docs disagree, label as `POSSIBLE_CONFLICT` and include the differing claims.
5) Never invent facts or use knowledge outside the provided materials.

OUTPUT FORMAT (STRICT)
QUESTION_BREAKDOWN:
- ...

FACTS_FOUND:
- [fact: value | source]

COURSE_CODES:
- [CODE ### - title] OR NO_INFO_FOUND

CONTACT_DATA:
- [contact fact] OR NO_INFO_FOUND

MISSING_FIELDS:
- [field name]

SOURCE_SUMMARY:
- [doc id or title : short note]
"""

SYSTEM_PROMPT_B_V3 = """You are Agent B (Refiner) for the University of Illinois BLS Virtual Advisor (v3).

GOAL
- Turn Agent A's structured facts into a concise, warm, and conservative student-facing answer.

MANDATORY QA
1) Escape every dollar sign as \$.
2) Remove citation artifacts and tags.
3) Remove LaTeX/math syntax.
4) Do not expose internal labels; summarize as "What we can confirm" / "What we could not confirm" when needed.
5) If a fact is `NO_INFO_FOUND`, say: "I do not have that specific information from the provided materials. Please contact the BLS office at onlineBLS@illinois.edu." Do not invent facts.

TONE
- Professional, warm, empathetic. Avoid negative comparisons to other LAS majors; state factual distinctions only.

OUTPUT
- Markdown bullets or short paragraphs (2-6 bullets). Use bold for key facts.
"""

SYSTEM_PROMPT_SINGLE_V3 = """You are the Official Virtual Assistant for the BLS program (v3).

CONSTRAINTS
1. Escape dollar signs with \$.
2. Remove citation tags and LaTeX math.

RESPOND
- Start with a one-line summary, then 2-4 bullets of concrete facts. Add a brief empathetic sentence when appropriate.
- If a fact is missing from materials, say: "I do not have that specific information from the provided documents. Please contact the BLS office at onlineBLS@illinois.edu." 

TONE
- Warm, respectful, and factual. When comparing programs, avoid negative framing.
"""

# 20 representative questions selected across all major sections/use-cases.
QUESTIONS = [
    "What is the Bachelor of Liberal Studies (BLS) degree?",
    "Who is the BLS program designed for?",
    "How is BLS different from other LAS majors?",
    "What are the admission requirements?",
    "Can I transfer credits? How many?",
    "Is there a minimum GPA requirement?",
    "When are application deadlines?",
    "What concentrations are available?",
    "Are there required courses?",
    "How long does it take to complete the degree?",
    "What is the tuition cost?",
    "Is financial aid available for part-time students?",
    "Does the BLS program accept the U of I employee tuition waiver?",
    "How do I schedule an advising appointment?",
    "What is the GPA requirement to stay in good standing?",
    "Will this degree feel legitimate compared to traditional majors?",
    "Is BLS a good option if I'm returning to school after many years?",
    "How flexible is the program if I work full-time?",
    "Can BLS help me pivot industries?",
    "How do employers view interdisciplinary degrees?",
]


def call_api(system_prompt: str, user_content: str, model: str = MODEL, timeout: int = 180) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "api_key": API_KEY,
        "course_name": COURSE_NAME,
        "stream": False,
        "temperature": 0.1,
        "retrieval_only": False,
    }

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == attempts:
                return {"ok": False, "message": f"API_REQUEST_FAILED: {exc}", "usage": {}, "status": 0}
            time.sleep(0.5 * attempt)
            continue

        if response.status_code == 200:
            try:
                body = response.json()
                return {
                    "ok": True,
                    "message": body.get("message", body.get("result", response.text)),
                    "usage": body.get("usage", {}),
                    "status": 200,
                }
            except ValueError:
                return {"ok": True, "message": response.text or "", "usage": {}, "status": 200}

        if response.status_code in (403, 404):
            return {"ok": False, "message": f"API_ERROR_{response.status_code}: {response.text}", "usage": {}, "status": response.status_code}

        if 500 <= response.status_code < 600 and attempt < attempts:
            time.sleep(0.5 * attempt)
            continue

        return {"ok": False, "message": f"API_ERROR_{response.status_code}: {response.text}", "usage": {}, "status": response.status_code}

    return {"ok": False, "message": "UNKNOWN_ERROR", "usage": {}, "status": 0}


def shorten(text: str, limit: int = 220) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def run() -> None:
    if not API_KEY:
        raise RuntimeError("Missing UIUC_CHAT_API_KEY in .env")
    if not COURSE_NAME:
        raise RuntimeError("Missing COURSE_NAME in .env")

    rows = []
    for idx, q in enumerate(QUESTIONS, start=1):
        print(f"[{idx}/{len(QUESTIONS)}] {q}")

        # Single-agent: use Refiner prompt directly for user-facing answer quality.
        t0 = time.time()
        single = call_api(SYSTEM_PROMPT_B, q)
        single_latency = round(time.time() - t0, 2)

        # Dual-agent: Retriever -> Refiner.
        t1 = time.time()
        draft = call_api(SYSTEM_PROMPT_A, q)
        draft_text = draft.get("message", "")
        refine_input = (
            f"Original User Query: {q}\n"
            f"Internal Draft Answer: {draft_text}\n\n"
            "Please refine this draft into a final response following your system constraints."
        )
        dual = call_api(SYSTEM_PROMPT_B, refine_input)
        dual_latency = round(time.time() - t1, 2)

        rows.append(
            {
                "question": q,
                "single": {
                    "answer": single.get("message", ""),
                    "ok": single.get("ok", False),
                    "latency": single_latency,
                    "usage": single.get("usage", {}),
                },
                "dual": {
                    "draft": draft_text,
                    "answer": dual.get("message", ""),
                    "ok": dual.get("ok", False) and draft.get("ok", False),
                    "latency": dual_latency,
                    "usage": {
                        "prompt_tokens": (draft.get("usage", {}).get("prompt_tokens", 0) or 0) + (dual.get("usage", {}).get("prompt_tokens", 0) or 0),
                        "completion_tokens": (draft.get("usage", {}).get("completion_tokens", 0) or 0) + (dual.get("usage", {}).get("completion_tokens", 0) or 0),
                        "total_tokens": (draft.get("usage", {}).get("total_tokens", 0) or 0) + (dual.get("usage", {}).get("total_tokens", 0) or 0),
                    },
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        time.sleep(1.0)

    out_json = RESULTS_DIR / "qwen_v2_20_results.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    md_lines = [
        "# Qwen 2.5 v2.0 Prompt Test (20 Representative Questions)",
        "",
        f"- Model: `{MODEL}`",
        "- Modes: Single-Agent (Refiner prompt directly), Dual-Agent (Retriever -> Refiner)",
        f"- API endpoint: `{API_URL}`",
        "",
        "| # | Question | Single-Agent Output Snippet | Single Latency (s) | Dual-Agent Output Snippet | Dual Latency (s) |",
        "| ---: | --- | --- | ---: | --- | ---: |",
    ]

    for i, row in enumerate(rows, start=1):
        md_lines.append(
            "| {i} | {q} | {sa} | {sl} | {da} | {dl} |".format(
                i=i,
                q=row["question"].replace("|", "\\|"),
                sa=shorten(row["single"]["answer"]).replace("|", "\\|"),
                sl=row["single"]["latency"],
                da=shorten(row["dual"]["answer"]).replace("|", "\\|"),
                dl=row["dual"]["latency"],
            )
        )

    out_md = PROJECT_DIR / "Qwen_v2_20_Run_Output.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nWrote JSON: {out_json}")
    print(f"Wrote Markdown: {out_md}")


def run_v3_two_questions() -> None:
    """Run only the two user-requested prompts through the v3 system prompts.

    Writes results to results/qwen_v3_two_questions.json and updates a short markdown.
    """
    two_qs = [
        "What is the Bachelor of Liberal Studies (BLS) degree?",
        "How is BLS different from other LAS majors?",
    ]

    rows = []
    for q in two_qs:
        print(f"Running v3 single-agent for: {q}")
        single = call_api(SYSTEM_PROMPT_SINGLE_V3, q)

        print(f"Running v3 dual-agent for: {q}")
        draft = call_api(SYSTEM_PROMPT_A_V3, q)
        draft_text = draft.get("message", "")
        refine_input = (
            f"Original User Query: {q}\n"
            f"Internal Draft Answer: {draft_text}\n\n"
            "Please refine this draft into a final response following your system constraints."
        )
        dual = call_api(SYSTEM_PROMPT_B_V3, refine_input)

        rows.append(
            {
                "question": q,
                "single": {"answer": single.get("message", ""), "ok": single.get("ok", False)},
                "dual": {"draft": draft_text, "answer": dual.get("message", ""), "ok": dual.get("ok", False) and draft.get("ok", False)},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        time.sleep(1.0)

    out_json = RESULTS_DIR / "qwen_v3_two_questions.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    md = PROJECT_DIR / "Qwen_v3_Two_Questions_Run.md"
    md_lines = [
        "# Qwen v3 Two-Question Run",
        "",
        f"- Model: `{MODEL}`",
        "- Prompts: v3 (updated system prompts)",
        "",
    ]
    for r in rows:
        md_lines.append(f"## {r['question']}")
        md_lines.append("\n**Single-Agent Answer:**\n")
        md_lines.append(r["single"]["answer"] or "(no answer)")
        md_lines.append("\n**Dual-Agent Draft:**\n")
        md_lines.append(r["dual"]["draft"] or "(no draft)")
        md_lines.append("\n**Dual-Agent Final:**\n")
        md_lines.append(r["dual"]["answer"] or "(no answer)")
        md_lines.append("---")

    md.write_text("\n\n".join(md_lines), encoding="utf-8")
    print(f"Wrote v3 JSON: {out_json}")
    print(f"Wrote v3 Markdown: {md}")


if __name__ == "__main__":
    import sys
    if "--v3-two" in sys.argv:
        run_v3_two_questions()
    else:
        run()


if __name__ == "__main__":
    run()
