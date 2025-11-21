from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[3]

# Data and storage paths
DATA_DIR = BASE_DIR / "backend" / "chatbot" / "rag" / "data"
FAQ_PATH = BASE_DIR / "docs" / "QA.json"
CHROMA_PERSIST_DIR = BASE_DIR / "backend" / "chatbot" / "rag" / "chroma_db"

# Embeddings model (local sentence-transformer)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FAQ similarity threshold (tune)
FAQ_SIM_THRESHOLD = 0.74

# Retriever / RAG settings
RETRIEVAL_TOP_K = 5  # Increased from 4 to get more context

OLLAMA_MODEL = "phi3:mini"

# System prompt — simplified and more direct for small models
SYSTEM_PROMPT = """You are FastBot, a helpful assistant for FAST-NUCES university.

IMPORTANT RULES:
1. Answer ONLY based on the context provided below
2. Be concise and direct
3. If the answer is not in the context, say: "I don't have that information in the documents provided"
4. Do not make up or invent any information
5. Use simple, clear language

Answer the user's question using the provided context."""