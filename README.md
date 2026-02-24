# BLS Chatbot

A production-ready, intelligent chatbot for the **Bachelor of Liberal Studies (BLS)** program at the University of Illinois. Built with a sophisticated two-agent architecture and powered by chat.illinois.edu's RAG (Retrieval Augmented Generation) platform.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Two-Agent System Explained](#two-agent-system-explained)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Cost](#cost)
- [Deployment](#deployment)
- [Support](#support)

## Overview

This chatbot provides prospective and current BLS students with accurate, well-formatted answers to their questions by:

1. **Retrieving information** from uploaded BLS program documents (syllabi, policies, FAQs, etc.)
2. **Grounding responses** in factual data to reduce AI hallucinations
3. **Formatting answers** in a student-friendly, professional manner
4. **Maintaining context** with fallback contact information

**Key Features:**

- ✅ **100% Free** - Uses NCSA-hosted Qwen/Qwen2.5-VL-72B-Instruct model
- ✅ **Two-Agent Architecture** - Separation of retrieval and refinement
- ✅ **Document Grounding** - All answers sourced from your uploaded materials
- ✅ **Production Ready** - Clean UI, error handling, proper formatting
- ✅ **Easy Deployment** - Flask backend, static HTML frontend

## Architecture

### System Flow Diagram

```
┌─────────────┐
│    User     │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         Flask Backend (app.py)          │
│  ┌───────────────────────────────────┐  │
│  │   Agent A: The Retriever          │  │
│  │   - Receives user query            │  │
│  │   - Calls chat.illinois.edu API    │  │
│  │   - Searches BLS documents         │  │
│  │   - Returns raw facts/data         │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│                 ▼                        │
│  ┌───────────────────────────────────┐  │
│  │   Agent B: The Refiner            │  │
│  │   - Takes Agent A's draft         │  │
│  │   - Formats for students          │  │
│  │   - Applies style rules           │  │
│  │   - Adds fallback contact info    │  │
│  │   - Returns polished answer       │  │
│  └──────────────┬────────────────────┘  │
└─────────────────┼────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Final Answer  │
         │   to User      │
         └────────────────┘
```

### Why Two Agents?

**Separation of Concerns** - Each agent has a single, focused responsibility:

1. **Agent A (Retriever)** focuses purely on **accuracy**
   - No formatting constraints
   - No tone requirements
   - Just extract facts from documents

2. **Agent B (Refiner)** focuses purely on **presentation**
   - Professional tone
   - Proper markdown formatting
   - Escape special characters
   - Add contact information when needed

This approach produces **better results** than a single-agent system because:

- The retrieval agent can focus on finding ALL relevant information
- The refinement agent can focus on making it readable and professional
- Easier to debug and improve each component independently
- Follows software engineering best practices (Single Responsibility Principle)

## Technology Stack

### Backend

- **Flask** - Lightweight Python web framework
  - Handles HTTP requests from frontend
  - Routes for serving HTML and API endpoints
  - CORS enabled for cross-origin requests

- **LangChain** - LLM application framework
  - `RunnableLambda` - Wraps agents as executable functions
  - `RunnableSequence` - Chains Agent A → Agent B in pipeline
  - Provides clean abstraction for multi-step LLM workflows

- **Requests** - HTTP library
  - Makes API calls to chat.illinois.edu
  - Handles authentication and error responses

### Frontend

- **HTML5** - Structure
  - Semantic markup
  - Accessible form elements
  - Clean chat interface

- **CSS3** - Styling
  - UIUC brand colors (Blue #13294B, Orange #E84A27)
  - Responsive design
  - Message bubbles (user vs. bot)
  - Smooth animations

- **Vanilla JavaScript** - Interactivity
  - Fetch API for backend communication
  - DOM manipulation for message rendering
  - Enter key submission
  - Loading states

### External Services

- **chat.illinois.edu** - RAG Platform
  - Document storage and indexing
  - Vector similarity search
  - LLM inference (Qwen model)
  - Built by UIUC/NCSA team

- **Qwen/Qwen2.5-VL-72B-Instruct** - Language Model
  - 72 billion parameters
  - Vision-capable (multimodal)
  - Free NCSA hosting
  - High quality reasoning

## Two-Agent System Explained

### Agent A: The Retriever

**Role:** Internal fact-finding agent

**System Prompt:**

```python
SYSTEM_PROMPT_A = """
### IDENTITY & GOAL
You are the internal Retrieval Agent for the Bachelor of Liberal Studies (BLS) 
program at the University of Illinois. Your sole objective is to search the 
provided documents and extract all factual information relevant to the user's query.

### PROTOCOL
1. Extract facts, figures, and policies related to the query.
2. If the answer is not in the documents, state clearly: "NO_INFO_FOUND".
3. Do not worry about formatting, escaping characters, or tone. 
   Focus purely on accuracy and comprehensiveness.
4. If the user asks about contact info, address, or emails, explicitly retrieve those.
"""
```

**Example Input:** "What is the tuition cost?"

**Example Output (Draft):**
```
The tuition is $433 per credit hour for Illinois residents. 
Out-of-state students pay $650 per credit hour.
Students need 120 credits to graduate.
Contact: onlineBLS@illinois.edu
```

### Agent B: The Refiner

**Role:** Student-facing communication agent

**System Prompt Highlights:**

```python
SYSTEM_PROMPT_B = """
### CRITICAL TECHNICAL CONSTRAINTS
1. ESCAPE DOLLAR SIGNS: Use \\$ not $
2. NO CITATION TAGS: Remove <cite>, <source>, [1]
3. NO MATH BLOCKS: Write math in plain English

### IDENTITY & GOAL
You are the Official Virtual Assistant for the BLS program.
Provide accurate, concise, and helpful information.

### FORMATTING RULES
1. Markdown with **bold** and bullet points
2. Short paragraphs (2-3 sentences)
3. Professional, encouraging tone

### CRITICAL STATIC DATA (FALLBACK)
- Office Address: 112 English Building, 608 S Wright St, Urbana, IL 61801
- Email: onlineBLS@illinois.edu
- Admissions Page: lasonline.illinois.edu/programs/bls
"""
```

**Example Input:** (Agent A's draft from above)

**Example Output (Final):**
```markdown
**Tuition Costs for BLS Program**

For Illinois residents, tuition is \\$433 per credit hour. Out-of-state students 
pay \\$650 per credit hour. The program requires 120 credits to graduate.

Need more information? Contact the BLS office at onlineBLS@illinois.edu or visit 
lasonline.illinois.edu/programs/bls.
```

### Why This Works

1. **Agent A** doesn't worry about formatting → gets more complete information
2. **Agent B** receives structured data → can focus on presentation
3. **Fallback data** in Agent B ensures contact info is always available
4. **Specialized prompts** make each agent better at their specific task


## Quick Start

### Step 1: Set up chat.illinois.edu

1. Go to <https://chat.illinois.edu>
2. Create a new project or navigate to: `bls-chatbot-v2`
3. Upload your BLS program documents:
   - Admissions requirements
   - Tuition and fees
   - Course catalog
   - Program policies
   - FAQs
   - Contact information
4. Go to the **Materials** page
5. Click **"Generate API Key"**
6. Copy the generated key (starts with `uc_...`)

### Step 2: Configure the Application

Edit [app.py](app.py) and update the API key:

```python
API_KEY = "uc_your_actual_key_here"  # Replace with your key from Step 1
```

### Step 3: Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install flask flask-cors requests langchain-core
```

### Step 4: Run the Chatbot

```bash
python app.py
```

Open your browser and visit: <http://127.0.0.1:5000>

### Step 5: Test It!

Try asking:
- "What is the tuition cost?"
- "How do I apply to the BLS program?"
- "What is the contact email?"

## Configuration

### Required Settings

Edit these values in [app.py](app.py):

```python
# API Configuration
UIUC_API_URL = "https://chat.illinois.edu/api/chat-api/chat"
API_KEY = "uc_cc91bdd73f4b497da3f0b0c73915ff06"
COURSE_NAME = "bls-chatbot-v2"
MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
```

### Optional Customization

**Change System Prompts:**

You can modify `SYSTEM_PROMPT_A` or `SYSTEM_PROMPT_B` to:
- Add program-specific instructions
- Change tone or style
- Add additional fallback data
- Customize security rules

**Switch Models:**

Free options:
- `Qwen/Qwen2.5-VL-72B-Instruct` (current, vision-capable)
- `llama3.1:70b` (alternative free option)

Paid options (requires OpenAI API key):
- `gpt-4o-mini` (best quality/price, ~$0.0006 per query)
- `gpt-4o` (highest quality, ~$0.003 per query)

## Project Structure

```
bls-chatbot/
├── app.py                          # Flask backend application
│   ├── Configuration               # API keys, model selection
│   └── Two-Agent System            # Agent A → Agent B pipeline
│       ├── SYSTEM_PROMPT_A         # Retrieval agent prompt
│       ├── SYSTEM_PROMPT_B         # Refinement agent prompt
│       ├── call_uiuc_chat()        # API communication function
│       ├── agent_a_retriever()     # Fact retrieval logic
│       ├── agent_b_refiner()       # Response formatting logic
│       └── multi_agent_chain       # LangChain pipeline
├── frontend/
│   └── index.html                  # Chat user interface
│       ├── HTML Structure          # Semantic markup
│       ├── CSS Styling             # UIUC brand colors
│       └── JavaScript              # Fetch API, DOM manipulation
├── venv/                           # Python virtual environment
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment variable template
├── README.md                       # This file
└── uiuc-chat-documentation.md      # Comprehensive deployment guide
```

## How It Works

### Request Flow

1. **User submits question** via HTML form
2. **Frontend JavaScript** sends POST to `/api/ask`
3. **Flask backend** receives request
4. **LangChain pipeline** executes:
   
   ```python
   multi_agent_chain = RunnableLambda(agent_a_retriever) | RunnableLambda(agent_b_refiner)
   ```

5. **Agent A** executes:
   - Calls `call_uiuc_chat(SYSTEM_PROMPT_A, user_query)`
   - chat.illinois.edu searches BLS documents
   - Returns draft response with raw facts
   
6. **Agent B** executes:
   - Receives Agent A's draft
   - Calls `call_uiuc_chat(SYSTEM_PROMPT_B, refinement_prompt)`
   - Formats response for students
   - Applies style rules and adds contact info
   
7. **Flask** returns JSON response to frontend
8. **JavaScript** displays formatted answer in chat UI

### chat.illinois.edu RAG Process

When Agent A or B calls the API:

```python
{
  "model": "Qwen/Qwen2.5-VL-72B-Instruct",
  "messages": [...],
  "course_name": "bls-chatbot-v2",      # Your document collection
  "retrieval_only": False,              # Get LLM response + contexts
  "temperature": 0.1                    # Low randomness for consistency
}
```

Behind the scenes:
1. **Query Embedding** - User question converted to vector
2. **Vector Search** - Find top 80 similar document chunks
3. **Context Assembly** - Add documents to LLM prompt
4. **LLM Inference** - Qwen generates answer grounded in docs
5. **Response** - Returns answer with source citations

### Error Handling

The application handles:

- **403 Forbidden** - Invalid API key (shows setup instructions)
- **404 Not Found** - Course name doesn't exist
- **Empty query** - Returns error message
- **Network failures** - Catches and displays errors
- **No documents** - Agent A returns "NO_INFO_FOUND", Agent B provides fallback

## Cost

### Current Setup: 100% FREE

- **chat.illinois.edu Platform**: Free
- **Qwen/Qwen2.5-VL-72B-Instruct Model**: Free (NCSA-hosted)
- **Document Storage**: Free
- **API Calls**: Unlimited (free tier)

**Total Cost: $0.00/month** 🎉

### Alternative: Paid Models

If you switch to OpenAI models for better quality:

| Model | Input Cost | Output Cost | Avg Query Cost | 1K Queries/Mo |
|-------|-----------|-------------|----------------|---------------|
| GPT-4o-mini | $0.150/1M tokens | $0.600/1M tokens | ~$0.0006 | ~$0.60 |
| GPT-4o | $5.00/1M tokens | $15.00/1M tokens | ~$0.003 | ~$3.00 |

**Recommendation**: Start with free Qwen model, upgrade to GPT-4o-mini only if needed.

## Deployment

### Local Development

Current setup (already configured):

```bash
python app.py  # Runs on http://127.0.0.1:5000
```

### Production Deployment Options

#### Option 1: University Hosting

Contact UIUC IT to deploy on official `.illinois.edu` domain.

**Pros:**
- Official university infrastructure
- Trusted domain
- IT support

#### Option 2: Cloud Platforms

**Railway/Render** (~$5-10/month):
```bash
# 1. Connect GitHub repo
# 2. Set environment variables
# 3. Deploy (automatic HTTPS)
```

**Vercel** (Free tier):
- Great for frontend
- Serverless functions for API

#### Option 3: Self-Hosted Server

Requirements:
- Ubuntu/Debian server
- Python 3.8+
- Nginx (reverse proxy)
- SSL certificate (Let's Encrypt)

See [uiuc-chat-documentation.md](uiuc-chat-documentation.md) for detailed deployment guide.

### Production Checklist

Before deploying to production:

- [ ] Change `debug=True` to `debug=False` in app.py
- [ ] Set up environment variables for secrets
- [ ] Enable CORS only for your domain
- [ ] Add rate limiting
- [ ] Set up error monitoring (Sentry)
- [ ] Configure logging
- [ ] Test on mobile devices
- [ ] Add analytics (optional)
- [ ] Create backup of documents
- [ ] Document admin procedures

## Advanced Usage

### Streaming Responses

For real-time streaming (like ChatGPT):

```python
data = {
    "stream": True,  # Enable streaming
    # ... other params
}

response = requests.post(url, headers=headers, json=data, stream=True)
for chunk in response.iter_lines():
    if chunk:
        print(chunk.decode())
```

### Using Images (Vision)

Qwen/Qwen2.5-VL-72B-Instruct supports image inputs:

```python
"messages": [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this diagram?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }
]
```

### Custom Tools

chat.illinois.edu supports custom tool use via N8N workflows:
- Calendar integration
- Application status checker
- Email notifications
- Database queries

See documentation for setup.

## Troubleshooting

### Common Issues

**"403 - Invalid API key"**
- Generate new key from chat.illinois.edu Materials page
- Update `API_KEY` in app.py
- Restart Flask server

**"404 - Course not found"**
- Verify `COURSE_NAME` matches your project name exactly
- Check spelling and case sensitivity

**Empty responses**
- Upload documents to chat.illinois.edu
- Verify documents are indexed (check Materials page)
- Try asking more specific questions

**Frontend not loading**
- Check Flask is running on port 5000
- Verify `frontend/index.html` exists
- Check browser console for errors

## Contributing

To improve this chatbot:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

Suggested improvements:
- Conversation history/memory
- User authentication
- Analytics dashboard
- Multi-language support
- Mobile app

## License

This project is for University of Illinois internal use. Check with your department for licensing requirements.

## Support

### Technical Support

- **chat.illinois.edu Platform**: rohan13@illinois.edu
- **BLS Program Information**: onlineBLS@illinois.edu

### Documentation

- [chat.illinois.edu Documentation](https://docs.uiuc.chat)
- [uiuc-chat-documentation.md](uiuc-chat-documentation.md) - Detailed setup guide
- [LangChain Documentation](https://python.langchain.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## Acknowledgments

- **UIUC NCSA** - For hosting free LLM infrastructure
- **chat.illinois.edu Team** - For the RAG platform
- **BLS Program** - For project support

---

**Built with ❤️ for the University of Illinois BLS Program**
