#!/usr/bin/env python3
"""
generate_docs.py — Create BLS_Chatbot_Model_Testing_SINGLE_AGENT.md
                   and BLS_Chatbot_Model_Testing_DUAL_AGENT.md
                   from saved test result JSON files.

Usage:
    python generate_docs.py
    python generate_docs.py --variant improved   # generates *_IMPROVED docs
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml

PIPELINE_DIR = Path(__file__).parent
PROJECT_DIR  = PIPELINE_DIR.parent
RESULTS_DIR  = PIPELINE_DIR / "results"
CONFIG_FILE  = PIPELINE_DIR / "config_models.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def get_answer(results: dict, config_id: str, q_key: str) -> str:
    """Retrieve the answer string for a given architecture and question key."""
    arch = results.get(config_id, {})
    entry = arch.get(q_key, {})
    answer = entry.get("answer", "")
    if not answer:
        return "_No response captured._"
    return answer.strip()


def meta_line(results: dict, config_id: str, q_key: str) -> str:
    """Return a small latency/token metadata line."""
    entry = results.get(config_id, {}).get(q_key, {})
    parts = []
    latency = entry.get("latency")
    if latency is not None:
        parts.append(f"⏱ {latency}s")
    usage = entry.get("usage", {})
    if usage:
        total = usage.get("total_tokens") or usage.get("totalTokens")
        if total:
            parts.append(f"🔢 {total} tokens")
    if entry.get("error"):
        parts.append("⚠️ error")
    return "  " + " | ".join(parts) if parts else ""


def section_anchor(section: str) -> str:
    return section.lower().replace(" ", "-").replace("/", "").replace("&", "")


# ---------------------------------------------------------------------------
# Single-Agent Document
# ---------------------------------------------------------------------------

def generate_single_agent_doc(
    results: dict,
    config_list: list,
    questions_sections: list,
    variant_tag: str = "",
) -> str:
    timestamp = datetime.now().strftime("%B %d, %Y %H:%M")
    lines = [
        f"# BLS Chatbot Model Testing — Single-Agent Architectures{variant_tag}",
        "",
        f"*Generated: {timestamp}*",
        "",
        "## Models Tested",
        "",
    ]
    for c in config_list:
        lines.append(f"- **{c['display_name']}** (`{c['model']}`)")
    lines += ["", "---", ""]

    # Table of contents
    lines.append("## Table of Contents")
    lines.append("")
    for section_obj in questions_sections:
        sec = section_obj["section"]
        lines.append(f"- [{sec}](#{section_anchor(sec)})")
        for sub in section_obj["subsections"]:
            if sub["name"]:
                lines.append(f"  - [{sub['name']}](#{section_anchor(sub['name'])})")
    lines += ["", "---", ""]

    # Content
    for section_obj in questions_sections:
        sec = section_obj["section"]
        lines.append(f"## {sec}")
        lines.append("")

        for sub in section_obj["subsections"]:
            subname = sub["name"]
            if subname:
                lines.append(f"### {subname}")
                lines.append("")

            for q in sub["questions"]:
                q_key = f"{sec}|{subname}|{q['num']}"
                lines.append(f"#### Q{q['num']}. {q['text']}")
                lines.append("")

                for arch in config_list:
                    answer = get_answer(results, arch["config_id"], q_key)
                    meta = meta_line(results, arch["config_id"], q_key)
                    lines.append(f"**{arch['display_name']} Response**")
                    if meta:
                        lines.append(f"*{meta.strip()}*")
                    lines.append("")
                    lines.append(answer)
                    lines.append("")

                lines.append("---")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dual-Agent Document
# ---------------------------------------------------------------------------

def generate_dual_agent_doc(
    results: dict,
    config_list: list,
    questions_sections: list,
    variant_tag: str = "",
) -> str:
    timestamp = datetime.now().strftime("%B %d, %Y %H:%M")
    lines = [
        f"# BLS Chatbot Model Testing — Dual-Agent Architectures{variant_tag}",
        "",
        f"*Generated: {timestamp}*",
        "",
        "## Architectures Tested",
        "",
    ]
    for c in config_list:
        lines.append(
            f"- **{c['display_name']}**  "
            f"Retriever: `{c['retriever_model']}` → Refiner: `{c['refiner_model']}`"
        )
    lines += ["", "---", ""]

    # Table of contents
    lines.append("## Table of Contents")
    lines.append("")
    for section_obj in questions_sections:
        sec = section_obj["section"]
        lines.append(f"- [{sec}](#{section_anchor(sec)})")
        for sub in section_obj["subsections"]:
            if sub["name"]:
                lines.append(f"  - [{sub['name']}](#{section_anchor(sub['name'])})")
    lines += ["", "---", ""]

    # Content
    for section_obj in questions_sections:
        sec = section_obj["section"]
        lines.append(f"## {sec}")
        lines.append("")

        for sub in section_obj["subsections"]:
            subname = sub["name"]
            if subname:
                lines.append(f"### {subname}")
                lines.append("")

            for q in sub["questions"]:
                q_key = f"{sec}|{subname}|{q['num']}"
                lines.append(f"#### Q{q['num']}. {q['text']}")
                lines.append("")

                for arch in config_list:
                    answer = get_answer(results, arch["config_id"], q_key)
                    meta = meta_line(results, arch["config_id"], q_key)
                    lines.append(
                        f"**{arch['display_name']}**"
                    )
                    if meta:
                        lines.append(f"*{meta.strip()}*")
                    lines.append("")
                    lines.append(answer)
                    lines.append("")

                lines.append("---")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate BLS model testing documents")
    parser.add_argument(
        "--variant",
        choices=["original", "improved"],
        default="original",
        help="Which result set to document (default: original)",
    )
    args = parser.parse_args()

    variant_tag = f"_{args.variant}" if args.variant != "original" else ""
    doc_suffix  = " (Improved Prompts)" if args.variant == "improved" else ""

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    from questions import QUESTION_SECTIONS  # noqa: E402

    sa_results_file = RESULTS_DIR / f"single_agent_results{variant_tag}.json"
    da_results_file = RESULTS_DIR / f"dual_agent_results{variant_tag}.json"

    sa_results = load_results(sa_results_file)
    da_results = load_results(da_results_file)

    sa_configs = config["architectures"]["single_agent"]
    da_configs = config["architectures"]["dual_agent"]

    # --- Single-agent document ---
    sa_doc = generate_single_agent_doc(
        sa_results, sa_configs, QUESTION_SECTIONS, doc_suffix
    )
    sa_out = PROJECT_DIR / f"BLS_Chatbot_Model_Testing_SINGLE_AGENT{variant_tag.upper()}.md"
    sa_out.write_text(sa_doc, encoding="utf-8")
    print(f"Written: {sa_out}")

    # --- Dual-agent document ---
    da_doc = generate_dual_agent_doc(
        da_results, da_configs, QUESTION_SECTIONS, doc_suffix
    )
    da_out = PROJECT_DIR / f"BLS_Chatbot_Model_Testing_DUAL_AGENT{variant_tag.upper()}.md"
    da_out.write_text(da_doc, encoding="utf-8")
    print(f"Written: {da_out}")

    print("\nNext step: python evaluate_responses.py")


if __name__ == "__main__":
    main()
