# backend/chatbot/rag/chain.py
import os
import json
import numpy as np
from transformers import pipeline
from .data_loader import load_and_split_docs
from .embeddings import get_embeddings_model
from .retriever import create_or_load_db
from .config import DATA_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL, QA_MODEL
from .website_loader import fetch_fast_content

# Globals (singletons per Python process)
_vectordb = None
_qa_pipeline = None
_embedding_model = None   # LangChain HuggingFaceEmbeddings object
_questions = []           # list of predefined QA questions (from QA.json if present)
_answers = []             # corresponding answers
_question_embeddings = None  # numpy array (N x D) normalized

def _l2_normalize(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / (norm + 1e-12)

def initialize_rag(force_reload=False):
    """
    Initialize / preload:
      - embedding model
      - persistent vectordb (from PDF splits)
      - QA pipeline (HF QA model)
      - precomputed question embeddings from QA.json (if exists)
    Call once at startup (or let views call it on first request).
    """
    global _vectordb, _qa_pipeline, _embedding_model, _questions, _answers, _question_embeddings

    if not force_reload and _qa_pipeline is not None and _vectordb is not None and _embedding_model is not None:
        return  # already initialized

    # 1) Embedding model (LangChain wrapper)
    print("🔹 Loading embedding model...")
    _embedding_model = get_embeddings_model(EMBEDDING_MODEL)

    # 2) Build/load vector DB for PDF docs
    try:
        print("🔹 Building/Loading vector DB from PDF/data...")
        splits = load_and_split_docs(DATA_PATH)
        _vectordb = create_or_load_db(splits, _embedding_model, VECTOR_DB_PATH)
    except Exception as exc:
        print(f"[chain.initialize_rag] vectordb load failed: {exc}")
        _vectordb = None

    # 3) QA pipeline (transformers)
    try:
        print(f"🔹 Loading QA model: {QA_MODEL}")
        _qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=-1)
    except Exception as exc:
        print(f"[chain.initialize_rag] QA pipeline load failed: {exc}")
        _qa_pipeline = None

    # 4) Load predefined QA.json (if exists) and precompute embeddings
    try:
        # Try both PDF and JSON possibilities, prefer QA.json in docs if present.
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # rag -> chatbot -> backend
        json_path = os.path.join(base_dir, "docs", "QA.json")
        if not os.path.exists(json_path):
            # sometimes docs path is up one level
            json_path = os.path.join(base_dir, "..", "docs", "QA.json")
            json_path = os.path.abspath(json_path)

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f)
            _questions = [p.get("question", "").strip() for p in qa_pairs]
            _answers = [p.get("answer", "").strip() for p in qa_pairs]

            # embed predefined questions
            if _embedding_model is not None and len(_questions) > 0:
                print("🔹 Embedding predefined QA questions...")
                raw = _embedding_model.embed_documents(_questions)  # list of vectors
                arr = np.array(raw, dtype=float)
                # normalize rows
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                _question_embeddings = arr / norms
            else:
                _question_embeddings = None
        else:
            _questions = []
            _answers = []
            _question_embeddings = None
    except Exception as exc:
        print(f"[chain.initialize_rag] loading QA.json failed: {exc}")
        _questions = []
        _answers = []
        _question_embeddings = None

    print("✅ RAG initialization complete.\n")

def _search_predefined(question, threshold=0.6):
    """
    Check predefined QA.json for a close match using embedding similarity.
    Returns answer string or None.
    """
    global _question_embeddings, _questions, _answers, _embedding_model

    if _question_embeddings is None or _embedding_model is None or len(_questions) == 0:
        return None

    try:
        q_emb = np.array(_embedding_model.embed_query(question), dtype=float)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        scores = np.dot(_question_embeddings, q_emb)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # debug print
        print(f"[predefined] best_score={best_score:.4f}, matched='{_questions[best_idx]}'")

        if best_score >= threshold:
            return _answers[best_idx]
    except Exception as exc:
        print(f"[chain._search_predefined] failed: {exc}")
    return None

def _search_pdf_rag(question, k=3):
    """
    Use vector DB (Chroma) to find top-k document chunks from PDFs and answer using QA model.
    Returns answer string or None.
    """
    global _vectordb, _qa_pipeline
    if _vectordb is None or _qa_pipeline is None:
        return None

    try:
        docs = _vectordb.similarity_search(question, k=k)
        if not docs:
            return None
        context = " ".join([doc.page_content for doc in docs])
        result = _qa_pipeline(question=question, context=context)
        # transformers QA pipeline returns dict with 'answer' and sometimes 'score'
        ans = result.get("answer")
        return ans
    except Exception as exc:
        print(f"[chain._search_pdf_rag] failed: {exc}")
        return None

def _split_text_into_chunks(text, chunk_size=800, chunk_overlap=200):
    """
    Simple character-based splitter (similar to your existing splitter).
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def _search_website(question, k=3):
    """
    Fetch FAST website content, chunk it, compute embeddings on chunks,
    find top-k most similar chunks and run QA pipeline on combined context.
    """
    global _embedding_model, _qa_pipeline
    if _embedding_model is None or _qa_pipeline is None:
        return None

    try:
        web_text = fetch_fast_content(max_chars=20000)
        if not web_text:
            return None

        chunks = _split_text_into_chunks(web_text, chunk_size=1000, chunk_overlap=150)
        if not chunks:
            return None

        # embed the chunks
        chunk_vecs = _embedding_model.embed_documents(chunks)  # list of vectors
        arr = np.array(chunk_vecs, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr_norm = arr / norms

        q_emb = np.array(_embedding_model.embed_query(question), dtype=float)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

        scores = np.dot(arr_norm, q_emb)
        top_idx = np.argsort(scores)[-k:][::-1]
        top_chunks = [chunks[i] for i in top_idx if scores[i] > 0.0]  # only keep positive similarities

        if not top_chunks:
            return None

        context = " ".join(top_chunks)
        result = _qa_pipeline(question=question, context=context)
        return result.get("answer")
    except Exception as exc:
        print(f"[chain._search_website] failed: {exc}")
        return None

def ask_question(query, predefined_threshold=0.65):
    """
    High-level function to answer a user query using:
      1) Predefined QA.json
      2) PDF RAG (vectordb + QA model)
      3) FAST website scraping + QA
    Returns the best-found answer string.
    """
    # ensure initialization
    initialize_rag()

    # 1) Predefined QA dataset
    ans = _search_predefined(query, threshold=predefined_threshold)
    if ans:
        return ans

    # 2) PDF RAG
    ans = _search_pdf_rag(query, k=3)
    if ans:
        return ans

    # 3) FAST website
    ans = _search_website(query, k=3)
    if ans:
        return ans

    # fallback
    return "Sorry, I couldn't find an authoritative answer. Please check the FAST website or ask an admin."
