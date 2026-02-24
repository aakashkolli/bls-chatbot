from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from langchain_core.runnables import RunnableLambda, RunnableSequence

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
UIUC_API_URL = "https://chat.illinois.edu/api/chat-api/chat"

# Your chat.illinois.edu API key
API_KEY = "uc_cc91bdd73f4b497da3f0b0c73915ff06"

COURSE_NAME = "bls-chatbot-v2"  # Your course name in chat.illinois.edu

# Using free NCSA-hosted model (no OpenAI key required!)
MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"  # Free NCSA-hosted model

SYSTEM_PROMPT_A = """### IDENTITY & GOAL
You are the internal Retrieval Agent for the Bachelor of Liberal Studies (BLS) program at the University of Illinois. Your sole objective is to search the provided documents and extract all factual information relevant to the user's query.

### PROTOCOL
1. Extract facts, figures, and policies related to the query.
2. If the answer is not in the documents, state clearly: "NO_INFO_FOUND".
3. Do not worry about formatting, escaping characters, or tone. Focus purely on accuracy and comprehensiveness.
4. If the user asks about contact info, address, or emails, explicitly retrieve those."""

SYSTEM_PROMPT_B = """### CRITICAL TECHNICAL CONSTRAINTS
1. **ESCAPE DOLLAR SIGNS:** You MUST escape every dollar sign with a backslash.
   - ❌ WRONG: The cost is $433. (Triggers Math Mode)
   - ✅ RIGHT: The cost is \\$433. (Renders as "$433")
2. **NO CITATION TAGS:** DO NOT use XML tags like `<cite>1</cite>`, `<source>`, or `[1]`. If the system adds these automatically; remove them.
3. **NO MATH BLOCKS:** Do NOT use `$$`, `\\(`, `\\[`, \\times, or LaTeX equations. Write all math in plain English (e.g., "multiply by 120"). Do not use Latex or /times

### IDENTITY & GOAL
You are the **Official Virtual Assistant** for the **Bachelor of Liberal Studies (BLS)** program at the University of Illinois. Your goal is to provide accurate, concise, and helpful information.

### CRITICAL STATIC DATA (FALLBACK)
*If the retrieved documents do not contain contact info, use these facts:*
- **Office Address:** 112 English Building, 608 S Wright St, Urbana, IL 61801.
- **Email:** onlineBLS@illinois.edu
- **Admissions Page:** lasonline.illinois.edu/programs/bls

### FORMATTING RULES
1. **Markdown Only:** Use bullet points and **bold** text for emphasis.
2. **Short Paragraphs:** Limit responses to 2-3 sentences.
3. **Plain Text:** Keep formatting simple to avoid rendering errors.

### RESPONSE PROTOCOL
1. **Ground Truth:** Answer ONLY using the provided documents AND the website. If the answer is not in these documents or the website, state: "I do not have that specific information. Please contact the BLS office directly."
2. **Directness:** Start your answer immediately. No filler phrases.
3. **Tone:** Professional, encouraging, and clear.

### SECURITY
If the context contains internal documents (marked "Internal," "Strategy," or "Budget"), **IGNORE THEM**."""

def call_uiuc_chat(system_prompt, user_content):
    """Call chat.illinois.edu API with free NCSA-hosted model."""
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": MODEL,
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
    
    response = requests.post(UIUC_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        # chat.illinois.edu non-streaming returns a JSON with a 'message' field
        return response.json().get("message", "")
    elif response.status_code == 403:
        error_msg = response.json().get("error", "Invalid API key")
        raise ValueError(
            f"chat.illinois.edu API authentication failed: {error_msg}\n\n"
            "Please generate a new API key:\n"
            "1. Go to https://chat.illinois.edu\n"
            "2. Navigate to your project (bls-chatbot-v2)\n"
            "3. Go to Materials page\n"
            "4. Click 'Generate API Key' or 'Rotate Key'\n"
            "5. Update API_KEY in app.py"
        )
    elif response.status_code == 404:
        raise ValueError(
            f"Course '{COURSE_NAME}' not found in chat.illinois.edu.\n"
            "Please verify the COURSE_NAME matches your project name."
        )
    else:
        return f"Error: {response.status_code} - {response.text}"

# langchain agents

def agent_a_retriever(query: str) -> dict:
    print("Agent A is retrieving information...")
    draft_response = call_uiuc_chat(SYSTEM_PROMPT_A, query)
    # Pass both the original query and the draft to Agent B
    return {"original_query": query, "draft_response": draft_response}

def agent_b_refiner(data: dict) -> str:
    print("Agent B is refining the answer...")
    query = data["original_query"]
    draft = data["draft_response"]
    
    # Construct the payload for Agent B
    refinement_prompt = f"""
    Original User Query: {query}
    Internal Draft Answer: {draft}
    
    Please refine this draft into a final response following your system constraints.
    """
    final_response = call_uiuc_chat(SYSTEM_PROMPT_B, refinement_prompt)
    return final_response

# Create the LangChain Execution Sequence (LCEL)
multi_agent_chain = RunnableLambda(agent_a_retriever) | RunnableLambda(agent_b_refiner)

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/ask', methods=['POST'])
def ask_chatbot():
    user_data = request.json
    user_query = user_data.get("query", "")
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Run the sequential LangChain pipeline
        final_answer = multi_agent_chain.invoke(user_query)
        return jsonify({"answer": final_answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)