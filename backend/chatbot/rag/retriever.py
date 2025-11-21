from langchain_community.vectorstores import Chroma

from langchain.schema import Document
from typing import List, Tuple
from chromadb.config import Settings
from pathlib import Path
import os

def build_or_load_chroma(docs: List[Document], persist_dir: str, embeddings):
    """
    Build or load a Chroma DB stored at persist_dir.
    If the DB already exists, this will create/append; for idempotency you
    may prefer to clear and rebuild when docs change.
    """
    persist_dir = str(Path(persist_dir).resolve())
    # Chroma settings: local-only
    client_settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    #db.persist()
    return db

def retrieve_docs(db: Chroma, query: str, k: int = 4):
    """
    Return list of (Document, score) from Chroma similarity search.
    """
    results = db.similarity_search_with_score(query, k=k)
    return results  # list of tuples (Document, score)

"""def create_or_load_db(splits, embedding_model, persist_dir):
    print("🔹 Creating/Loading vector database...")
    vectordb = Chroma.from_documents(splits, embedding_model, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb  """
