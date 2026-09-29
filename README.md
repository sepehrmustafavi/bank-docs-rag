# Bank Docs RAG

A simple prototype of a Retrieval-Augmented Generation (RAG) pipeline for
answering questions over bank documents, using open-source models and ChromaDB.

## Stack
- LLM: Llama 3.1 8B Instruct (via Ollama)
- Embeddings: intfloat/multilingual-e5-base
- Vector store: ChromaDB
- Sample documents are synthetic (no real bank data)

## How it works
1. `ingest.py` loads txt/pdf files, splits them into overlapping chunks,
   embeds them and stores them in ChromaDB.
2. `query.py` embeds the question, retrieves the top-k chunks and asks
   the LLM to answer only from that context.

## Setup
    ollama pull llama3.1:8b
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Usage
    python src/ingest.py
    python src/query.py "حداکثر مبلغ وام مسکن چقدر است؟"
    python src/query.py            # interactive mode

## Limitations / Future work
- Character-based chunking (could use semantic/recursive chunking)
- No reranker
- No evaluation set yet