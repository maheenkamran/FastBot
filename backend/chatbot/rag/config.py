import os

# Get the absolute path to the project root (the main FastBot folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__))))))

# File paths
DATA_PATH = os.path.join(BASE_DIR, "docs", "QA.pdf")             # You can change to QA.json or QA.txt later
VECTOR_DB_PATH = os.path.join(BASE_DIR, "backend", "chatbot", "rag", "chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"