# Agentic RAG Chatbot — MVP (RAG + Logs)

Prototype RAG chatbot for a **German PDF product catalog**:
- Ingests `./data/*.pdf`
- Extracts text + tables
- Chunks and indexes content in a local vector DB (Chroma)
- Answers questions in **English** with **citations (page + snippet)**
- Logs each pipeline step for observability (timings + key artifacts)

## Features
- **End-to-end pipeline:** PDF → extraction → chunking → embeddings → retrieval → answer
- **Table-aware extraction** (catalogs often store Art.-Nr / PZN / sizes in tables)
- **English answers + citations** (page numbers + short snippets)
- **Observability:** JSONL step logs in `./logs/run.jsonl`
- **Optional LLM (Groq):**
  - Improves query rewrite / reranking / final answer fluency when enabled
  - Fully works without any LLM (offline-first fallback)

## Repo Structure
