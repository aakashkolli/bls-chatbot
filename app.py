from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import time
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
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

# v3: Softer, more explicit about uncertainty and decomposition
SYSTEM_PROMPT_A_V3 = """You are Agent A (Retriever) for the University of Illinois BLS Virtual Advisor (v3).

PURPOSE
- Extract precise, document-grounded facts to support a student-facing answer. Prefer conservative, source-backed claims.

RETRIEVAL & DECOMPOSITION
1) Decompose the user query into explicit sub-questions and list them verbatim.
2) For each sub-question, return one of: a short factual value, `NO_INFO_FOUND`, or `POSSIBLE_CONFLICT` with the conflicting values documented.
3) When returning dates or program status, avoid guessing launch/availability; if not present, return `NO_INFO_FOUND`.
4) If a fact is derived from multiple documents, include a `SOURCE_SUMMARY` line with document ids/titles.
5) Never invent facts or use outside knowledge not present in the provided documents.

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
- [doc id or title : short note]
"""

# v3: Softer tone, empathy scaffold, and updated guardrails
SYSTEM_PROMPT_B_V3 = """You are Agent B (Refiner) for the University of Illinois BLS Virtual Advisor (v3).

GOAL
- Convert Agent A's structured fact pack into a concise, student-facing response that is warm, factual, and cautious about uncertainty.

MANDATORY QA
1) Escape every dollar sign as \\$.
2) Remove citation artifacts and tags: <cite>, </cite>, <source>, [1], [2], etc.
3) Remove LaTeX and math syntax: $$, \\(...\\), \\[...\\], \\times.
4) Do not reveal internal labels (QUESTION_BREAKDOWN, FACTS_FOUND) to the end user; instead rewrite them into a short "What we can confirm" / "What we could not confirm" summary when needed.
5) If a required fact is `NO_INFO_FOUND`, say: "I do not have that specific information from the materials provided. Please contact the BLS office at onlineBLS@illinois.edu." Do not invent alternatives.

TONE & STYLE
- Professional and warm (think: helpful advisor). Include a brief empathetic sentence when answering subjective/concerned questions (e.g., "I understand this can feel uncertain — here's what we know.").
- Avoid negative or comparative language about other LAS majors; focus on distinctions without denigration.

OUTPUT FORMAT
- Markdown bullets or short paragraphs (2-6 bullets). Use bold for key facts. Keep answers concise but complete.

FALLBACK CONTACT
- If contact info is missing from the documents, include: onlineBLS@illinois.edu

"""


def call_uiuc_chat(system_prompt, user_content, model=None):
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
                return js.get("message", js.get("result", response.text))
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

# langchain agents

def call_retriever(query: str, retriever_model: str = None) -> dict:
    """Execute Agent A retrieval and return structured payload for Agent B."""
    draft_response = call_uiuc_chat(SYSTEM_PROMPT_A_V3, query, model=retriever_model)
    return {"original_query": query, "draft_response": draft_response}


def call_refiner(data: dict, refiner_model: str = None) -> str:
    """Execute Agent B refinement on Agent A output."""
    query = data["original_query"]
    draft = data["draft_response"]
    model_to_use = refiner_model if refiner_model is not None else data.get("refiner_model", None)
    refinement_prompt = f"""
Original User Query: {query}
Internal Draft Answer: {draft}

Please refine this draft into a final response following your system constraints.
"""
    return call_uiuc_chat(SYSTEM_PROMPT_B_V3, refinement_prompt, model=model_to_use)


def return_to_user(refined_answer: str) -> str:
    """Final response adapter for outbound API response consistency."""
    return refined_answer


def agent_a_retriever(query: str) -> dict:
    print("Agent A is retrieving information...")
    return call_retriever(query)

def agent_b_refiner(data: dict) -> str:
    print("Agent B is refining the answer...")
    return call_refiner(data)


def run_dual_agent(query: str, retriever_model: str = None, refiner_model: str = None) -> str:
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)