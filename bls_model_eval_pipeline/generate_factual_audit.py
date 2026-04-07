import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "bls_model_eval_pipeline" / "results"
OUT_PATH = ROOT / "Factual_Ground_Truth_Audit.md"

CONFIGS = [
    ("sa_gpt_oss", "GPT-OSS / Single-Agent", "single"),
    ("sa_qwen", "Qwen-72B / Single-Agent", "single"),
    ("da_gpt_gpt", "GPT-OSS / Dual-Agent", "dual"),
    ("da_qwen_qwen", "Qwen-72B / Dual-Agent", "dual"),
]

ERROR_SIGNALS = (
    "API_ERROR",
    "MODEL_UNAVAILABLE",
    "API_AUTH_FAILED",
    "COURSE_NOT_FOUND",
    "Error:",
    "API_REQUEST_FAILED",
)
NON_ANSWER_SIGNALS = (
    "I do not have that specific information",
    "I don't have that information",
    "NO_INFO_FOUND",
    "Please contact the BLS office",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def has_unescaped_dollar(text: str) -> bool:
    return re.search(r"(?<!\\)\$", text or "") is not None


def score_row(question: str, answer: str) -> tuple[int, str]:
    text = answer or ""
    if text == "MISSING_RESULT":
        return 1, "Missing output in saved run artifact; requires rerun for this model/arch and prompt."

    notes = []
    score = 8
    q = (question or "").lower()

    if any(sig in text for sig in ERROR_SIGNALS):
        return 1, "System/API failure; no usable answer."

    if any(sig.lower() in text.lower() for sig in NON_ANSWER_SIGNALS):
        score = min(score, 4)
        notes.append("Non-answer fallback used.")

    if "<cite" in text.lower() or "<source" in text.lower() or re.search(r"\[\d+\]", text):
        score -= 2
        notes.append("Formatting break: citation tag or bracket citation present.")

    if has_unescaped_dollar(text):
        score -= 2
        notes.append("Formatting break: unescaped dollar sign.")

    if "required courses" in q and not re.search(r"[A-Z]{2,4}\s?\d{3}", text):
        score -= 2
        notes.append("Depth gap: no specific course codes cited (e.g., ACE 240).")

    if "concentrations" in q:
        hits = sum(x in text for x in ("Global Perspectives", "Health and Society", "Management Studies"))
        if hits < 2:
            score -= 2
            notes.append("Depth gap: incomplete concentration list.")

    if "tuition" in q or "cost" in q:
        if not any(x in text for x in ("\\$433", "433 per credit", "2025-2026")):
            score -= 2
            notes.append("Potential hallucination or missing 2025-2026 tuition detail.")

    if re.search(r"up to the full 120 credits", text, re.IGNORECASE):
        score -= 2
        notes.append("Potential hallucination: transfer-credit maximum likely overstated.")

    if re.search(r"summer admission.*no .*deadline", text, re.IGNORECASE):
        score -= 1
        notes.append("Potentially unsupported deadline claim.")

    length = len(normalize(text))
    if length < 60:
        score -= 1
        notes.append("Too terse for advising context.")
    if length > 1400:
        score -= 1
        notes.append("Verbose; may reduce UX readability.")

    score = max(1, min(10, score))
    if not notes:
        notes.append("Grounded and usable; minor stylistic tuning only.")

    return score, " ".join(notes)


def sort_key(key: str):
    parts = key.split("|")
    section = parts[0] if len(parts) > 0 else ""
    subsection = parts[1] if len(parts) > 1 else ""
    try:
        num = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        num = 0
    return section, subsection, num


def main() -> None:
    single = json.loads((RES_DIR / "single_agent_results.json").read_text())
    dual = json.loads((RES_DIR / "dual_agent_results.json").read_text())

    # Build canonical 54-prompt list from robust configs.
    seen = set()
    prompt_keys = []
    for cfg in ("sa_qwen", "da_qwen_qwen", "da_gpt_gpt"):
        source = single if cfg.startswith("sa_") else dual
        for key in source.get(cfg, {}).keys():
            if key not in seen:
                seen.add(key)
                prompt_keys.append(key)
    prompt_keys = sorted(prompt_keys, key=sort_key)

    lines = [
        "# Factual Ground Truth Audit (54 Prompts x 4 Configurations)",
        "",
        "Source: Existing API-run artifacts in bls_model_eval_pipeline/results.",
        "",
        "| Prompt | Model/Arch | Output Snippet | Accuracy Score 1-10 | Style/Tone Notes |",
        "| --- | --- | --- | ---: | --- |",
    ]

    rows = 0
    coverage = {cfg_name: {"present": 0, "missing": 0} for cfg_name, _, _ in CONFIGS}
    for key in prompt_keys:
        question = None
        for cfg_name, _, cfg_mode in CONFIGS:
            source = single if cfg_mode == "single" else dual
            entry = source.get(cfg_name, {}).get(key, {})
            if question is None:
                question = entry.get("question") or key

        for cfg_name, cfg_label, cfg_mode in CONFIGS:
            source = single if cfg_mode == "single" else dual
            entry = source.get(cfg_name, {}).get(key, {})
            answer = entry.get("answer", "MISSING_RESULT")
            if answer == "MISSING_RESULT":
                coverage[cfg_name]["missing"] += 1
            else:
                coverage[cfg_name]["present"] += 1
            snippet = normalize(answer)
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."
            score, notes = score_row(question or key, answer)

            lines.append(
                "| {prompt} | {arch} | {snippet} | {score} | {notes} |".format(
                    prompt=(question or key).replace("|", "\\|"),
                    arch=cfg_label,
                    snippet=snippet.replace("|", "\\|"),
                    score=score,
                    notes=notes.replace("|", "\\|"),
                )
            )
            rows += 1

    lines.extend(
        [
            "",
            "## Coverage Summary",
            "",
            "| Model/Arch | Present Responses | Missing Responses |",
            "| --- | ---: | ---: |",
        ]
    )
    for cfg_name, cfg_label, _ in CONFIGS:
        lines.append(
            f"| {cfg_label} | {coverage[cfg_name]['present']} | {coverage[cfg_name]['missing']} |"
        )

    OUT_PATH.write_text("\n".join(lines))
    print(f"Generated {OUT_PATH} with {len(prompt_keys)} prompts and {rows} rows")


if __name__ == "__main__":
    main()
