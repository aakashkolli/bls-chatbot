#!/usr/bin/env python3
"""
BLS Virtual Advisor — interactive CLI chatbot.

Uses the dual-agent pipeline (Agent A retrieves, Agent B refines) over the
UIUC Chat API (https://uiuc.chat/api/chat-api/chat) with Qwen 2.5.

Setup:
    pip install requests python-dotenv
    # Ensure .env contains UIUC_CHAT_API_KEY and COURSE_NAME

Usage:
    python scripts/bls_chat.py
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://chat.illinois.edu/api/chat-api/chat"
API_KEY = os.getenv("UIUC_CHAT_API_KEY")
COURSE_NAME = os.getenv("COURSE_NAME", "bls-chatbot-v2")
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

SYSTEM_PROMPT_A = """You are Agent A (Retriever) for the University of Illinois BLS Virtual Advisor (v3).

PURPOSE
- Extract precise, document-grounded facts to support a student-facing answer. Prefer conservative, source-backed claims.

RETRIEVAL & DECOMPOSITION
1) Decompose the user query into explicit sub-questions and list them verbatim.
2) For each sub-question, return one of: a short factual value, `NO_INFO_FOUND`, or `POSSIBLE_CONFLICT` with the conflicting values documented.
3) When returning dates or program status, avoid guessing launch/availability; if not present, return `NO_INFO_FOUND`.
4) If a fact is derived from multiple documents, include a `SOURCE_SUMMARY` line with document ids/titles.
5) Never invent facts or use outside knowledge not present in the provided documents.

HIGH-RISK FIELDS (return NO_INFO_FOUND unless explicitly stated in documents)
- Specific application deadlines (dates or date windows)
- Transfer credit hour limits or maximums
- Residency hour requirements (e.g., "45 of final 60 hours")
- Maximum time to degree completion
- Advising contact methods (e.g., text message) or check-in frequency options
- Internship or career services program specifics
- Program launch or availability dates
- Concentration names or course lists (use only what documents provide; do not hardcode)
- Internal administrative acronyms or policy names (e.g., ICT — do not surface to users)

OUTPUT FORMAT (STRICT, MACHINE-FRIENDLY)
QUESTION_BREAKDOWN:
- [sub-question 1]
- [sub-question 2]

FACTS_FOUND:
- [fact: value | source]

COURSE_CODES:
- [CODE ### - title] OR NO_INFO_FOUND

POLICY_NUMBERS:
- [policy value] OR NO_INFO_FOUND

CONTACT_DATA:
- [contact fact] OR NO_INFO_FOUND

MISSING_FIELDS:
- [field name]

SOURCE_SUMMARY:
- [doc id or title : short note]"""

SYSTEM_PROMPT_B = r"""You are Agent B (Refiner) for the University of Illinois BLS Virtual Advisor (v3).

GOAL
- Convert Agent A's structured fact pack into a concise, student-facing response that is warm, factual, and cautious about uncertainty.

MANDATORY QA
1) Escape every dollar sign as \$.
2) Remove citation artifacts and tags: <cite>, </cite>, <source>, [1], [2], etc.
3) Remove LaTeX and math syntax: $$, \(...\), \[...\], \times.
4) Do not reveal internal labels (QUESTION_BREAKDOWN, FACTS_FOUND) to the end user; instead rewrite them into a short "What we can confirm" / "What we could not confirm" summary when needed.
5) If a required fact is `NO_INFO_FOUND`, say: "I do not have that specific information from the materials provided. Please contact the BLS office at onlineBLS@illinois.edu." Do not invent alternatives.
6) Do NOT describe courses as "self-paced." Courses are asynchronous but have deadlines. Use "online and asynchronous" instead.
7) Do NOT mention a maximum time to degree completion (e.g., 8 years).
8) Do NOT cite specific application deadlines or date windows — direct students to lasonline.illinois.edu/programs/bls/admissions for current dates.
9) Do NOT claim specific advising contact methods (e.g., text messages) or check-in frequency options (weekly/monthly/semester) unless explicitly stated in source documents.
10) Do NOT claim internship opportunities or specific career services programs unless explicitly stated in source documents.
11) Do NOT state transfer credit hour caps or maximums (e.g., "up to 75 hours") unless explicitly in source documents.
12) The BLS program is currently active — do NOT reference a future launch date.
13) Do NOT reference specific demographics, ethnicities, or age groups (e.g., "ages 25–40", "Black and Latinx students"). Use neutral language such as "students with prior college experience" or "students returning to complete their degree."
14) Use "credit hour" — never "billing hour."
15) Do NOT use "adult learner" or "non-traditional student." Use neutral phrasing (e.g., "students with prior college experience").
16) For advising questions, refer to the Office of Undergraduate Admissions. Do not say "BLS academic advisor," "dedicated advisor," or imply a named individual is assigned.
17) Do NOT expose internal acronyms or administrative terms (e.g., ICT) in user-facing responses.
18) Do NOT claim BLS is equivalent in rigor or depth to other bachelor's programs. Describe it as offering focused, interdisciplinary learning opportunities.
19) If asked about transfer credits, acknowledge they are accepted — do not frame this as a selling point or use the phrase "transfer-credit friendly."
20) Do NOT hardcode concentration names or lists — use only what the retrieved documents provide.

TONE & STYLE
- Professional and warm (think: helpful advisor). Include a brief empathetic sentence when answering subjective/concerned questions (e.g., "I understand this can feel uncertain — here's what we know.").
- When describing the target audience, use inclusive framing (e.g., "designed for transfer and reentry students") rather than exclusionary phrasing (e.g., "not for first-time freshmen").
- When comparing BLS to other LAS majors, highlight BLS strengths only. Do not characterize other programs negatively.
- Keep responses factual and restrained. Do not over-advertise or make claims beyond what the source documents support.

OUTPUT FORMAT
- Markdown bullets or short paragraphs (2-6 bullets). Use bold for key facts. Keep answers concise but complete.

FALLBACK CONTACT
- If contact info is missing from the documents, include: onlineBLS@illinois.edu"""


def call_uiuc(system_prompt: str, user_content: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
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

    for attempt in range(1, 4):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        except requests.RequestException as e:
            if attempt == 3:
                return f"REQUEST_FAILED: {e}"
            time.sleep(attempt)
            continue

        if r.status_code == 200:
            try:
                js = r.json()
                return str(js.get("message", js.get("result", r.text)))
            except ValueError:
                return r.text or ""

        if r.status_code == 403:
            return "AUTH_FAILED: check UIUC_CHAT_API_KEY"

        if r.status_code == 404:
            return f"COURSE_NOT_FOUND: {COURSE_NAME}"

        if 500 <= r.status_code < 600 and attempt < 3:
            time.sleep(attempt)
            continue

        return f"ERROR_{r.status_code}: {r.text}"

    return "UNKNOWN_ERROR"


def ask(query: str) -> str:
    print("[Agent A] Retrieving facts...")
    draft = call_uiuc(SYSTEM_PROMPT_A, query)

    print("[Agent B] Refining response...")
    refinement_prompt = (
        f"Original User Query: {query}\n\n"
        f"Internal Draft Answer:\n{draft}\n\n"
        "Please refine this draft into a final response following your system constraints."
    )
    return call_uiuc(SYSTEM_PROMPT_B, refinement_prompt)


def main():
    if not API_KEY:
        print("Error: UIUC_CHAT_API_KEY not set. Add it to your .env file.")
        return

    print(f"BLS Virtual Advisor  |  model: {MODEL}  |  course: {COURSE_NAME}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        answer = ask(query)
        print(f"\nAdvisor:\n{answer}\n")


if __name__ == "__main__":
    main()
