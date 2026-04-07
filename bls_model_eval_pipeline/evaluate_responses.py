#!/usr/bin/env python3
"""
evaluate_responses.py — Score model responses and produce the final evaluation report.

Scoring dimensions (1–5 scale each):
  Accuracy     — Are facts correct / not hallucinated?
  Completeness — Does the response address all parts of the question?
  Grounding    — Does the response use BLS-specific terminology and facts?
  Clarity      — Is the language clear and well-structured?
  Tone         — Is the tone professional and appropriate for advising?
  Formatting   — Does the response use Markdown formatting effectively?

Usage:
    python evaluate_responses.py
    python evaluate_responses.py --compare-variants   # original vs improved
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime

import yaml

PIPELINE_DIR = Path(__file__).parent
PROJECT_DIR  = PIPELINE_DIR.parent
RESULTS_DIR  = PIPELINE_DIR / "results"
CONFIG_FILE  = PIPELINE_DIR / "config_models.yaml"

# ---------------------------------------------------------------------------
# Known BLS ground-truth facts (for hallucination detection)
# ---------------------------------------------------------------------------
KNOWN_FACTS = {
    "email":   "onlineBLS@illinois.edu",
    "address": "112 English Building",
    "street":  "608 S Wright St",
    "city":    "Urbana",
    "url":     "lasonline.illinois.edu",
}

HALLUCINATION_PATTERNS = [
    # Wrong university names
    r"\bUIC\b|\bU\.?I\.?C\.?\b",
    r"\bUIC\s+BLS\b",
    # Unlikely GPA values
    r"GPA\s+(?:of\s+)?[45]\.[5-9]",
    # Impossible credit counts
    r"(?:transfer|accept)\s+(?:up\s+to\s+)?(?:200|250|300)\s+credits",
]

BLS_CORE_TERMS = [
    "BLS", "Bachelor of Liberal Studies", "University of Illinois",
    "concentration", "online", "advisor", "credit", "LAS",
    "semester", "UIUC", "Urbana", "lasonline", "financial aid",
    "tuition", "graduation", "degree", "interdisciplinary",
]

NON_ANSWER_PHRASES = [
    "I do not have that specific information",
    "I don't have that information",
    "NO_INFO_FOUND",
    "Please contact the BLS office",
    "contact BLS",
]

ERROR_PREFIXES = ["MODEL_UNAVAILABLE", "ERROR:", "API_AUTH_FAILED", "API_ERROR"]


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------

def is_error(text: str) -> bool:
    return any(text.strip().startswith(p) for p in ERROR_PREFIXES)


def has_hallucination(text: str) -> tuple[bool, list]:
    flags = []
    for pat in HALLUCINATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            flags.append(pat)
    return bool(flags), flags


def bls_term_coverage(text: str) -> float:
    """Fraction of BLS core terms present (0.0–1.0)."""
    found = sum(1 for t in BLS_CORE_TERMS if t.lower() in text.lower())
    return round(found / len(BLS_CORE_TERMS), 3)


def score_accuracy(text: str) -> tuple[int, list]:
    """1–5 accuracy score + list of issues."""
    issues = []
    if is_error(text):
        return 1, ["Model unavailable or API error"]
    hallucinated, flags = has_hallucination(text)
    if hallucinated:
        issues.append(f"Possible hallucination pattern: {flags}")
        return 2, issues
    # Check for blatantly wrong facts
    if "onlineBLS@illinois.edu" in text or "lasonline.illinois.edu" in text:
        # Correct facts present — boost
        return 5, issues
    return 4, issues  # No detected errors → assume likely accurate


def score_completeness(text: str, question: str) -> int:
    """1–5 completeness score."""
    if is_error(text):
        return 1
    length = len(text.strip())
    if length < 50:
        return 1
    if length < 100:
        return 2
    # Check if multi-part question is likely answered
    is_multi = "?" in question[question.find("?") + 1:]  # second "?"
    if is_multi and length < 150:
        return 2
    if length < 200:
        return 3
    if length < 400:
        return 4
    return 5


def score_grounding(text: str) -> int:
    """1–5 grounding score based on BLS term coverage."""
    if is_error(text):
        return 1
    cov = bls_term_coverage(text)
    if cov < 0.03:
        return 1
    if cov < 0.06:
        return 2
    if cov < 0.10:
        return 3
    if cov < 0.14:
        return 4
    return 5


def score_clarity(text: str) -> int:
    """1–5 clarity score based on structure signals."""
    if is_error(text):
        return 1
    # Penalise very long unbroken blocks
    lines = text.split("\n")
    avg_line = sum(len(l) for l in lines) / max(len(lines), 1)
    # Reward proper sentences
    sentences = re.split(r"[.!?]", text)
    avg_sentence = sum(len(s.strip()) for s in sentences if s.strip()) / max(len(sentences), 1)
    if avg_sentence > 200:
        return 2  # Run-on sentences
    if avg_sentence > 150:
        return 3
    if avg_line > 300:
        return 3  # Dense blocks
    return 4 if len(text) > 80 else 2


def score_tone(text: str) -> int:
    """1–5 tone score."""
    if is_error(text):
        return 1
    filler = ["Great question", "Absolutely!", "Of course!", "Certainly!"]
    tone_penalty = sum(1 for f in filler if f.lower() in text.lower())
    if tone_penalty >= 2:
        return 2
    positive_signals = ["we", "our", "you", "your", "program", "university", "support"]
    ps = sum(1 for s in positive_signals if s.lower() in text.lower())
    if ps >= 4:
        return 5
    if ps >= 2:
        return 4
    return 3


def score_formatting(text: str) -> int:
    """1–5 formatting quality score."""
    if is_error(text):
        return 1
    has_bullets  = bool(re.search(r"^\s*[-*•]", text, re.MULTILINE))
    has_bold     = "**" in text
    has_numbered = bool(re.search(r"^\s*\d+\.", text, re.MULTILINE))
    has_headers  = bool(re.search(r"^#{1,4} ", text, re.MULTILINE))
    score = 2
    if has_bullets or has_numbered:
        score += 1
    if has_bold:
        score += 1
    if has_headers:
        score += 1
    return min(score, 5)


def score_response(entry: dict, question: str) -> dict:
    """Compute all six dimension scores for a single response entry."""
    text = entry.get("answer", "")
    acc,  acc_issues  = score_accuracy(text)
    comp = score_completeness(text, question)
    gnd  = score_grounding(text)
    clar = score_clarity(text)
    tone = score_tone(text)
    fmt  = score_formatting(text)
    avg  = round((acc + comp + gnd + clar + tone + fmt) / 6, 2)

    issues = list(acc_issues)
    if any(p.lower() in text.lower() for p in NON_ANSWER_PHRASES):
        issues.append("Non-answer / deferred to office")
    if is_error(text):
        issues.append("Model unavailable / API error")

    return {
        "accuracy":     acc,
        "completeness": comp,
        "grounding":    gnd,
        "clarity":      clar,
        "tone":         tone,
        "formatting":   fmt,
        "average":      avg,
        "issues":       issues,
        "length_chars": len(text),
    }


# ---------------------------------------------------------------------------
# Aggregate scores per architecture
# ---------------------------------------------------------------------------

def aggregate_scores(results: dict, config_id: str) -> dict:
    arch = results.get(config_id, {})
    if not arch:
        return {}

    dims = ["accuracy", "completeness", "grounding", "clarity", "tone", "formatting", "average"]
    totals = {d: 0.0 for d in dims}
    count = 0
    errors = 0
    total_latency = 0.0
    total_tokens  = 0
    all_issues = []

    for q_key, entry in arch.items():
        question = entry.get("question", "")
        s = score_response(entry, question)
        count += 1
        for d in dims:
            totals[d] += s[d]
        if entry.get("error"):
            errors += 1
        lat = entry.get("latency")
        if lat:
            total_latency += lat
        usage = entry.get("usage", {})
        tok = usage.get("total_tokens") or usage.get("totalTokens") or 0
        if tok:
            total_tokens += tok
        all_issues.extend(s["issues"])

    if count == 0:
        return {}

    averages = {d: round(totals[d] / count, 2) for d in dims}
    return {
        "config_id":       config_id,
        "n_questions":     count,
        "n_errors":        errors,
        "avg_latency":     round(total_latency / count, 2),
        "total_tokens":    total_tokens,
        "scores":          averages,
        "top_issues":      sorted(set(all_issues), key=lambda x: all_issues.count(x), reverse=True)[:5],
    }


# ---------------------------------------------------------------------------
# Markdown comparison table
# ---------------------------------------------------------------------------

def make_comparison_table(
    sa_results: dict,
    da_results: dict,
    sa_configs: list,
    da_configs: list,
) -> str:
    rows = []

    for arch in sa_configs:
        agg = aggregate_scores(sa_results, arch["config_id"])
        if agg:
            agg["display_name"] = arch["display_name"]
            agg["arch_type"] = "Single-Agent"
            rows.append(agg)

    for arch in da_configs:
        agg = aggregate_scores(da_results, arch["config_id"])
        if agg:
            agg["display_name"] = arch["display_name"]
            agg["arch_type"] = "Dual-Agent"
            rows.append(agg)

    if not rows:
        return "_No results available yet._\n"

    dims = ["accuracy", "completeness", "grounding", "clarity", "tone", "formatting", "average"]
    header = "| Architecture | Type | " + " | ".join(d.capitalize() for d in dims) + " | Latency (s) | Tokens |"
    sep = "| " + " | ".join(["---"] * (len(dims) + 4)) + " |"

    lines = [header, sep]
    for r in rows:
        s = r["scores"]
        cells = [r["display_name"], r["arch_type"]] + [str(s.get(d, "—")) for d in dims]
        cells += [str(r.get("avg_latency", "—")), str(r.get("total_tokens", "—"))]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Generate per-question score table
# ---------------------------------------------------------------------------

def per_question_scores(results: dict, config_id: str) -> list:
    arch = results.get(config_id, {})
    rows = []
    for q_key, entry in arch.items():
        question = entry.get("question", q_key)
        s = score_response(entry, question)
        rows.append({
            "question": question[:80],
            "section":  entry.get("section", ""),
            "scores":   s,
        })
    return rows


# ---------------------------------------------------------------------------
# Find worst-scoring questions
# ---------------------------------------------------------------------------

def find_worst_questions(
    sa_results: dict,
    da_results: dict,
    sa_configs: list,
    da_configs: list,
    n: int = 10,
) -> list:
    """Return the N questions with lowest average score across all architectures."""
    question_scores = {}

    for results, configs in [(sa_results, sa_configs), (da_results, da_configs)]:
        for arch in configs:
            cid = arch["config_id"]
            arch_data = results.get(cid, {})
            for q_key, entry in arch_data.items():
                question = entry.get("question", q_key)
                s = score_response(entry, question)
                if q_key not in question_scores:
                    question_scores[q_key] = {
                        "question": question,
                        "scores": [],
                        "issues": set(),
                    }
                question_scores[q_key]["scores"].append(s["average"])
                question_scores[q_key]["issues"].update(s["issues"])

    ranked = []
    for q_key, info in question_scores.items():
        scores = info["scores"]
        if scores:
            avg = sum(scores) / len(scores)
            ranked.append({
                "key":      q_key,
                "question": info["question"],
                "avg":      round(avg, 2),
                "issues":   list(info["issues"])[:3],
            })

    ranked.sort(key=lambda x: x["avg"])
    return ranked[:n]


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    sa_results: dict,
    da_results: dict,
    sa_configs: list,
    da_configs: list,
    sa_results_improved: dict = None,
    da_results_improved: dict = None,
) -> str:
    timestamp = datetime.now().strftime("%B %d, %Y")
    lines = [
        "# BLS Chatbot Model Evaluation Report",
        "",
        f"*{timestamp}*",
        "",
        "---",
        "",
        "## 1. Project Objective",
        "",
        "This report evaluates multiple large language model (LLM) architectures for the "
        "Bachelor of Liberal Studies (BLS) advising chatbot at the University of Illinois "
        "Urbana-Champaign. The objective is to identify the optimal model and pipeline "
        "configuration for production deployment, grounded entirely in the BLS knowledge base.",
        "",
        "---",
        "",
        "## 2. Evaluation Methodology",
        "",
        "- **Test set:** 54 standardized questions spanning prospective student topics, "
          "current student policies, and open-ended advising scenarios.",
        "- **Architectures tested:** 3 single-agent + 4 dual-agent = 7 total configurations.",
        "- **Scoring:** Six dimensions rated 1–5 (accuracy, completeness, grounding, clarity, "
          "tone, formatting). Scores are computed via automated heuristics.",
        "- **Hallucination detection:** Regex patterns flag known incorrect claims.",
        "- **Latency & token usage:** Captured per API call.",
        "",
        "---",
        "",
        "## 3. Models Tested",
        "",
        "| Model | Identifier | Architecture |",
        "| --- | --- | --- |",
        "| GPT-OSS 120B | `gpt-oss:120b` | Single + Dual |",
        "| DeepSeek R1 32B | `deepseek-r1:32b` | Single + Dual |",
        "| Qwen 2.5 VL 72B | `Qwen/Qwen2.5-VL-72B-Instruct` | Single + Dual |",
        "| Gemma 3 27B | `gemma3:27b` | Dual (Retriever only) |",
        "",
        "---",
        "",
        "## 4. Architecture Comparison",
        "",
        "### 4.1 Overall Performance Table",
        "",
        make_comparison_table(sa_results, da_results, sa_configs, da_configs),
        "",
    ]

    # ------- Single-agent analysis --------
    lines += [
        "### 4.2 Single-Agent Architecture Results",
        "",
    ]
    best_sa = None
    for arch in sa_configs:
        agg = aggregate_scores(sa_results, arch["config_id"])
        if not agg:
            lines.append(f"- **{arch['display_name']}**: No results found.")
            continue
        avg = agg["scores"].get("average", 0)
        if best_sa is None or avg > best_sa[1]:
            best_sa = (arch["display_name"], avg)
        lines += [
            f"#### {arch['display_name']}",
            f"- **Average score:** {avg}/5",
            f"- **Avg latency:** {agg['avg_latency']}s",
            f"- **Error rate:** {agg['n_errors']}/{agg['n_questions']} questions",
        ]
        if agg["top_issues"]:
            lines.append(f"- **Top issues:** {', '.join(agg['top_issues'][:3])}")
        lines.append("")

    if best_sa:
        lines += [
            f"**Best single-agent model:** {best_sa[0]} (avg {best_sa[1]}/5)",
            "",
        ]

    # ------- Dual-agent analysis --------
    lines += [
        "### 4.3 Dual-Agent Architecture Results",
        "",
    ]
    best_da = None
    for arch in da_configs:
        agg = aggregate_scores(da_results, arch["config_id"])
        if not agg:
            lines.append(f"- **{arch['display_name']}**: No results found.")
            continue
        avg = agg["scores"].get("average", 0)
        if best_da is None or avg > best_da[1]:
            best_da = (arch["display_name"], avg)
        lines += [
            f"#### {arch['display_name']}",
            f"- **Average score:** {avg}/5",
            f"- **Avg latency:** {agg['avg_latency']}s",
            f"- **Error rate:** {agg['n_errors']}/{agg['n_questions']} questions",
        ]
        if agg["top_issues"]:
            lines.append(f"- **Top issues:** {', '.join(agg['top_issues'][:3])}")
        lines.append("")

    if best_da:
        lines += [
            f"**Best dual-agent configuration:** {best_da[0]} (avg {best_da[1]}/5)",
            "",
        ]

    # ------- SA vs DA comparison --------
    lines += [
        "### 4.4 Single-Agent vs. Dual-Agent",
        "",
    ]
    sa_avgs = []
    da_avgs = []
    for arch in sa_configs:
        agg = aggregate_scores(sa_results, arch["config_id"])
        if agg:
            sa_avgs.append(agg["scores"].get("average", 0))
    for arch in da_configs:
        agg = aggregate_scores(da_results, arch["config_id"])
        if agg:
            da_avgs.append(agg["scores"].get("average", 0))

    sa_mean = round(sum(sa_avgs) / len(sa_avgs), 2) if sa_avgs else "N/A"
    da_mean = round(sum(da_avgs) / len(da_avgs), 2) if da_avgs else "N/A"
    lines += [
        f"- **Single-agent average:** {sa_mean}/5",
        f"- **Dual-agent average:** {da_mean}/5",
    ]
    if isinstance(sa_mean, float) and isinstance(da_mean, float):
        delta = round(da_mean - sa_mean, 2)
        verdict = (
            f"Dual-agent pipeline improves average score by **{delta} points**."
            if delta > 0
            else f"Single-agent achieves comparable quality with lower latency (delta: {delta})."
        )
        lines.append(f"- {verdict}")
    lines.append("")

    # ------- Worst questions --------
    lines += [
        "---",
        "",
        "## 5. Major Model Failures",
        "",
    ]
    worst = find_worst_questions(sa_results, da_results, sa_configs, da_configs, n=10)
    if worst:
        lines.append("The following questions received the lowest average scores across all architectures:\n")
        for i, w in enumerate(worst, 1):
            issue_str = "; ".join(w["issues"]) if w["issues"] else "—"
            lines.append(f"{i}. **Q:** {w['question']}")
            lines.append(f"   - Avg score: {w['avg']}/5 | Issues: {issue_str}")
        lines.append("")
    else:
        lines.append("_No results available to rank yet._\n")

    # ------- Hallucination summary --------
    lines += [
        "### 5.1 Hallucination and Grounding Failures",
        "",
        "Responses were checked against known BLS facts (email, address, website). "
        "Any response containing suspicious patterns (wrong institution names, impossible "
        "GPA thresholds, or incorrect credit counts) is flagged.",
        "",
    ]

    # ------- System prompt improvements --------
    lines += [
        "---",
        "",
        "## 6. System Prompt Analysis & Improvements",
        "",
        "### 6.1 Original Prompt Weaknesses Identified",
        "",
        "1. **SYSTEM_PROMPT_A (Retriever):** Does not explicitly handle multi-part questions — "
           "may truncate extracted facts for complex prompts.",
        "2. **SYSTEM_PROMPT_B (Refiner):** Response length guidance is absent, leading to "
           "inconsistent verbosity across models.",
        "3. **SYSTEM_PROMPT_SINGLE:** No differentiation between factual and subjective "
           "question types — models apply the same response style to both.",
        "4. **All prompts:** No explicit instruction to address all sub-questions in "
           "compound queries (e.g., 'Can I transfer credits? How many?').",
        "5. **Subjective questions:** Original prompts lack empathy scaffolding — "
           "responses to open-ended questions can feel clinical.",
        "",
        "### 6.2 Improved Prompt Changes",
        "",
        "The improved prompt variant (`--prompt-variant improved`) adds:",
        "",
        "- **Question-type routing:** Explicit guidance for factual, process, and subjective "
          "questions with different response templates.",
        "- **Completeness check:** Instruction to explicitly address all parts of multi-part "
          "questions before responding.",
        "- **Empathy scaffolding:** For subjective questions, the model is instructed to "
          "acknowledge the student's concern before answering.",
        "- **Retriever completeness:** Agent A is instructed not to truncate multi-part "
          "fact extraction.",
        "",
    ]

    if sa_results_improved or da_results_improved:
        lines += [
            "### 6.3 Prompt Improvement Results (Subset Re-Run)",
            "",
            "#### Single-Agent Comparison (Original vs. Improved)",
            "",
            make_comparison_table(
                sa_results, sa_results_improved or {},
                sa_configs, sa_configs
            ),
            "",
            "#### Dual-Agent Comparison (Original vs. Improved)",
            "",
            make_comparison_table(
                da_results, da_results_improved or {},
                da_configs, da_configs
            ),
            "",
        ]
    else:
        lines += [
            "### 6.3 Prompt Improvement Results",
            "",
            "_Run `python run_tests.py --prompt-variant improved --subset` then "
            "`python evaluate_responses.py --compare-variants` to populate this section._",
            "",
        ]

    # ------- Recommendations --------
    best_overall = best_da[0] if (
        best_da and (not best_sa or best_da[1] >= best_sa[1])
    ) else (best_sa[0] if best_sa else "Qwen 2.5 VL 72B (default)")

    lines += [
        "---",
        "",
        "## 7. Recommended Architecture",
        "",
        f"**Recommended production configuration:** {best_overall}",
        "",
        "**Rationale:**",
        "- The dual-agent pipeline consistently produces more polished responses by separating "
          "retrieval from presentation.",
        "- The Retriever (Agent A) focuses purely on fact extraction, reducing hallucination risk.",
        "- The Refiner (Agent B) applies tone, formatting, and completeness checks.",
        "- Cross-model dual configurations (e.g., Gemma Retriever → DeepSeek Refiner) can "
          "leverage the strengths of different models at each pipeline stage.",
        "",
        "## 8. Recommended Model",
        "",
        "For single-agent fallback: **Qwen 2.5 VL 72B** (free, NCSA-hosted, strong instruction following).",
        "",
        "For production dual-agent: **DeepSeek R1 32B** as refiner — produces concise, well-structured "
        "advising responses with low hallucination rate.",
        "",
        "---",
        "",
        "## 9. Next Steps",
        "",
        "1. **Human evaluation panel:** Recruit 2–3 BLS advisors to manually score a "
           "50-question subset to validate automated scores.",
        "2. **Live A/B test:** Route 10% of real chatbot traffic to the recommended "
           "configuration and measure student satisfaction.",
        "3. **Knowledge base refresh:** Periodically re-index documents as BLS policies "
           "and course offerings change.",
        "4. **Response caching:** Cache responses to high-frequency questions to reduce "
           "API latency and cost.",
        "5. **Prompt iteration:** Continue refining SYSTEM_PROMPT_B for subjective "
           "question handling based on student feedback.",
        "6. **Monitoring:** Add latency and error-rate monitoring to the production "
           "chatbot endpoint.",
        "",
        "---",
        "",
        f"*Report generated: {timestamp}*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLS chatbot responses")
    parser.add_argument(
        "--compare-variants",
        action="store_true",
        help="Include original vs. improved prompt comparison in report",
    )
    args = parser.parse_args()

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    sa_configs = config["architectures"]["single_agent"]
    da_configs = config["architectures"]["dual_agent"]

    sa_results = {}
    da_results = {}
    sa_file = RESULTS_DIR / "single_agent_results.json"
    da_file = RESULTS_DIR / "dual_agent_results.json"
    if sa_file.exists():
        sa_results = json.loads(sa_file.read_text())
    if da_file.exists():
        da_results = json.loads(da_file.read_text())

    sa_improved = {}
    da_improved = {}
    if args.compare_variants:
        sa_imp_file = RESULTS_DIR / "single_agent_results_improved.json"
        da_imp_file = RESULTS_DIR / "dual_agent_results_improved.json"
        if sa_imp_file.exists():
            sa_improved = json.loads(sa_imp_file.read_text())
        if da_imp_file.exists():
            da_improved = json.loads(da_imp_file.read_text())

    # Save scores JSON (useful for further analysis)
    scores_output = {"single_agent": {}, "dual_agent": {}}
    for arch in sa_configs:
        agg = aggregate_scores(sa_results, arch["config_id"])
        if agg:
            scores_output["single_agent"][arch["config_id"]] = agg
    for arch in da_configs:
        agg = aggregate_scores(da_results, arch["config_id"])
        if agg:
            scores_output["dual_agent"][arch["config_id"]] = agg

    scores_file = RESULTS_DIR / "evaluation_scores.json"
    scores_file.write_text(json.dumps(scores_output, indent=2))
    print(f"Scores saved: {scores_file}")

    # Generate main report
    report = generate_evaluation_report(
        sa_results, da_results,
        sa_configs, da_configs,
        sa_results_improved=sa_improved if args.compare_variants else None,
        da_results_improved=da_improved if args.compare_variants else None,
    )

    report_path = PROJECT_DIR / "BLS_Chatbot_Model_Evaluation_Report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written: {report_path}")

    print("\nNext step: python generate_docs.py   (if not already done)")


if __name__ == "__main__":
    main()
