# BLS Chatbot

A chatbot for the Bachelor of Liberal Studies (BLS) program at the University of Illinois.

## System Flow

```mermaid
graph TD
    User[User]
    User --> Retriever[Agent A: Retriever]
    Retriever -->|Extracts facts from documents| Chat[chat.illinois.edu API]
    Chat -->|Returns raw facts| Retriever
    Retriever -->|Raw facts| Refiner[Agent B: Refiner]
    Refiner -->|Refines, formats, adds fallback info| Flask[Flask Backend]
    Flask -->|Serves answer| UI[Frontend]
    UI -->|Displays answer| User
    Docs[BLS Documents]
    Chat --> Docs
    Refiner -->|Uses fallback contact info if needed| Fallback[Static Data]
    Fallback --> Refiner
```

## Two-Agent Approach

Two agents are used: one retrieves facts, the other refines and formats answers for clarity and accuracy.

## Tech Stack

- Python (Flask, LangChain)
- HTML, CSS, JavaScript
- Requests (API calls)

## External Services

- chat.illinois.edu: document-grounded answers, free NCSA-hosted LLM
