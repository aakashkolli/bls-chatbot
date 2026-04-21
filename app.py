from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import json
import re
import requests
import time
from typing import Any, Optional
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
UIUC_API_URL = "https://chat.illinois.edu/api/chat-api/chat"

import os
# Load sensitive config from environment
API_KEY = os.getenv("UIUC_CHAT_API_KEY")
COURSE_NAME = os.getenv("COURSE_NAME")
MODEL = os.getenv("MODEL", "gemini-2.5-pro-exp-03-25")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT_A_V3 = f"""You are Agent A (Retriever) for the University of Illinois BLS Virtual Advisor.

PURPOSE
- Extract precise, document-grounded facts only.
- Prioritize factual recall over polished writing.

RETRIEVAL APPROACH
1) Decompose the user query into explicit sub-questions.
2) For each sub-question, return one of: factual value, NO_INFO_FOUND, or POSSIBLE_CONFLICT with both claims.
3) Include only facts supported by provided materials; do not use outside knowledge.
4) For date-sensitive or frequently changing details (deadlines, key dates, concentration lists), avoid hardcoded values unless explicitly present in retrieved materials.
5) If a detail is not clearly supported, return NO_INFO_FOUND.

COMPLIANCE CONSTRAINTS
- Do not introduce demographic targeting language (age, ethnicity, DEI framing).
- Do not expose internal acronyms or internal administrative process labels.
- Do not infer program launch timelines.
- Do not infer transfer-credit limits or maximum time-to-completion values.

CANONICAL SOURCES FOR CURRENT DETAILS
- Admissions FAQ: https://lasonline.illinois.edu/programs/bls/admissions/#faq
- Program page: https://lasonline.illinois.edu/programs/bls/

OUTPUT FORMAT
QUESTION_BREAKDOWN:
- [sub-question]

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
- [doc id or title : short note]
"""

SYSTEM_PROMPT_B_V3 = f"""You are Agent B (Refiner) for the University of Illinois BLS Virtual Advisor.

GOAL
- Convert Agent A's fact pack into a concise, student-facing response that is warm, factual, and conservative.

RESPONSE RULES
1) Do not reveal internal labels (QUESTION_BREAKDOWN, FACTS_FOUND, etc.).
2) Remove citation artifacts and technical tags.
3) If a required fact is NO_INFO_FOUND, say clearly that the information is not available in current materials and provide a next step.
4) Use precise terminology: the program is online and asynchronous (not self-paced).
5) Do not mention maximum time-to-completion.
6) Do not hardcode deadlines or key dates; direct users to the Admissions FAQ for current details.
7) Do not hardcode concentration lists; rely on retrieved facts, or direct users to the program page for current offerings.
8) Use "credit hour" terminology.
9) Use neutral, inclusive language; avoid references to age, ethnicity, or demographic targeting.
10) Do not use "adult learner" or "non-traditional" phrasing.
11) For advising/scheduling questions, refer to the Office of Undergraduate Admissions (do not mention text messaging workflows or assigned personal advisors unless explicitly documented).
12) Do not expose internal acronyms or administrative details.
13) Do not claim equivalence in rigor/depth versus other degrees; describe BLS as focused, interdisciplinary learning.
14) If asked about transfer credit, acknowledge transfer credit is accepted when supported, without promotional framing.
15) Do not over-advertise; keep claims proportional to verified facts.

STYLE
- Professional, warm, restrained.
- Keep answers concise and user-facing.
- Prefer 2-6 bullets or a short paragraph + bullets.

REQUIRED LINKING FOR DYNAMIC DETAILS
- Admissions FAQ: https://lasonline.illinois.edu/programs/bls/admissions/#faq
- Program page: https://lasonline.illinois.edu/programs/bls/

FALLBACK CONTACT
- onlineBLS@illinois.edu
"""


def call_uiuc_chat(system_prompt: str, user_content: str, model: Optional[str] = None) -> str:
    """Call chat.illinois.edu API with free NCSA-hosted model.
    
    Args:
        system_prompt: The system instructions for the model.
        user_content: The user's query or message.
        model: Optional model override. Defaults to MODEL env variable.
    """
    _model = model if model is not None else MODEL
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": _model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "api_key": API_KEY,
        "course_name": COURSE_NAME,
        "stream": False,
        "temperature": 0.1,
        "retrieval_only": False
    }
    
    # Attempt the request with retries for transient network or server issues
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(UIUC_API_URL, headers=headers, json=data, timeout=30)
        except requests.RequestException as e:
            if attempt == attempts:
                return f"API_REQUEST_FAILED: {str(e)}"
            time.sleep(0.5 * attempt)
            continue

        # On 200, try to parse JSON but fallback to raw text if parsing fails
        if response.status_code == 200:
            try:
                js = response.json()
                # chat.illinois.edu non-streaming returns a JSON with a 'message' field
                return str(js.get("message", js.get("result", response.text)))
            except ValueError:
                # not JSON — return raw text so we still capture the model output
                return response.text or ""

        # Authentication error
        if response.status_code == 403:
            try:
                error_msg = response.json().get("error", "Invalid API key")
            except ValueError:
                error_msg = response.text or "Invalid API key"
            return f"API_AUTH_FAILED: {error_msg}"

        # Course not found
        if response.status_code == 404:
            return f"COURSE_NOT_FOUND: {COURSE_NAME}"

        # For other 5xx errors, retry a couple times
        if 500 <= response.status_code < 600 and attempt < attempts:
            time.sleep(0.5 * attempt)
            continue

        # Other non-200 responses — return the body for diagnostics
        try:
            text = response.json()
        except ValueError:
            text = response.text
        return f"Error: {response.status_code} - {text}"

    return "UNKNOWN_ERROR"


def call_gemini(system_prompt: str, user_content: str, model: str = "gemini-3-flash-preview", timeout: int = 30) -> str:
    """Call Google's Generative Language API (generateContent)."""
    api_key = GEMINI_API_KEY
    if not api_key:
        return "MISSING_GEMINI_API_KEY"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    payload = {"contents": [{"parts": [{"text": f"{system_prompt}\n\nQuestion: {user_content}\nAnswer:"}]}]}

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return f"API_REQUEST_FAILED: {exc}"

    if r.status_code != 200:
        try:
            j = r.json()
            return f"API_ERROR_{r.status_code}: {j}"
        except Exception:
            return f"API_ERROR_{r.status_code}: {r.text}"

    try:
        j = r.json()
    except ValueError:
        return r.text or ""

    text = None
    if isinstance(j, dict):
        candidates = j.get("candidates") or j.get("outputs") or j.get("responses") or j.get("results")
        if isinstance(candidates, list) and candidates:
            cand = candidates[0]
            content = cand.get("content") or cand.get("output") or cand.get("text")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    text = "\n".join(parts)
            elif isinstance(content, str):
                text = content

        if not text:
            for k in ("message", "output", "result", "text"):
                v = j.get(k)
                if isinstance(v, str):
                    text = v
                    break

    if not text:
        def find(obj):
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                for _, v in obj.items():
                    found = find(v)
                    if found:
                        return found
            if isinstance(obj, list):
                for it in obj:
                    found = find(it)
                    if found:
                        return found
            return None

        text = find(j) or json.dumps(j)

    return text


def call_model(system_prompt: str, user_content: str, model: Optional[str] = None) -> str:
    """Dispatch to Gemini or UIUC/NCSA-hosted models by model name."""
    _model = model if model is not None else MODEL
    if isinstance(_model, str) and _model.lower().startswith("gemini"):
        return call_gemini(system_prompt, user_content, model=_model)
    return call_uiuc_chat(system_prompt, user_content, model=_model)

# langchain agents

def call_retriever(query: str, retriever_model: Optional[str] = None) -> dict[str, Any]:
    """Execute Agent A retrieval and return structured payload for Agent B."""
    draft_response = call_model(SYSTEM_PROMPT_A_V3, query, model=retriever_model)
    return {"original_query": query, "draft_response": draft_response}


def call_refiner(data: dict[str, Any], refiner_model: Optional[str] = None) -> str:
    """Execute Agent B refinement on Agent A output."""
    query = data["original_query"]
    draft = data["draft_response"]
    model_to_use = refiner_model if refiner_model is not None else data.get("refiner_model", None)
    refinement_prompt = f"""
Original User Query: {query}
Internal Draft Answer: {draft}

Please refine this draft into a final response following your system constraints.
"""
    return call_model(SYSTEM_PROMPT_B_V3, refinement_prompt, model=model_to_use)


def return_to_user(refined_answer: str) -> str:
    """Final response adapter for outbound API response consistency."""
    return refined_answer


def agent_a_retriever(query: str) -> dict:
    print("Agent A is retrieving information...")
    return call_retriever(query)

def agent_b_refiner(data: dict[str, Any]) -> str:
    print("Agent B is refining the answer...")
    return call_refiner(data)


def run_dual_agent(query: str, retriever_model: Optional[str] = None, refiner_model: Optional[str] = None) -> str:
    """Run dual-agent pipeline with optional per-agent model overrides.
    
    Args:
        query: The user's question.
        retriever_model: Model for Agent A (retrieval). Defaults to MODEL env variable.
        refiner_model: Model for Agent B (refinement). Defaults to MODEL env variable.
    """
    print(f"Dual-agent run | retriever={retriever_model or MODEL} | refiner={refiner_model or MODEL}")
    draft_payload = call_retriever(query, retriever_model=retriever_model)
    refined = call_refiner(draft_payload, refiner_model=refiner_model)
    return return_to_user(refined)


# Create the LangChain Execution Sequence (LCEL)
multi_agent_chain = RunnableLambda(agent_a_retriever) | RunnableLambda(agent_b_refiner)

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/widget')
def widget():
    return send_from_directory('frontend', 'widget.html')

@app.route('/embed.js')
def embed_js():
    return send_from_directory('frontend', 'embed.js'), 200, \
        {'Content-Type': 'application/javascript'}


@app.route('/api/ask', methods=['POST'])
def ask_chatbot():
    user_data = request.json
    user_query = user_data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        print("Running in dual-agent mode...")
        final_answer = multi_agent_chain.invoke(user_query)
        return jsonify({"answer": final_answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat_widget_api():
    """Embeddable widget endpoint: call_retriever() -> call_refiner() -> return_to_user()."""
    user_data = request.json or {}
    query = user_data.get("query", "")
    retriever_model = user_data.get("retriever_model", None)
    refiner_model = user_data.get("refiner_model", None)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        draft_payload = call_retriever(query, retriever_model=retriever_model)
        refined_answer = call_refiner(draft_payload, refiner_model=refiner_model)
        final_answer = return_to_user(refined_answer)
        return jsonify({
            "answer": final_answer,
            "pipeline": "call_retriever -> call_refiner -> return_to_user"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def sanitize_user_response(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"</?(?:cite|source)>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&lt;/?(?:cite|source)&gt;", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+"Citation\s*\d+"', "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bCitation\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b#page=\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bp\.\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[\w\-]+\.(?:pdf|pptx|docx|txt)\b(?:,\s*p\.\d+)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\s*([,;:])", r"\2", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


@app.route('/api/stream', methods=['POST'])
def stream_chat():
    user_data = request.json or {}
    query = user_data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    def generate():
        yield f"data: {json.dumps({'type': 'agent_status', 'agent': 'A', 'message': 'Retrieving facts from BLS documents…'})}\n\n"
        draft_payload = call_retriever(query)

        yield f"data: {json.dumps({'type': 'agent_status', 'agent': 'B', 'message': 'Refining response for you…'})}\n\n"
        refined = call_refiner(draft_payload)
        final = sanitize_user_response(return_to_user(refined))

        yield f"data: {json.dumps({'type': 'answer', 'content': final})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)