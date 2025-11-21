# backend/chatbot/rag/chain.py
import os
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional

# Try to use langchain_ollama if available
try:
    from langchain_ollama import ChatOllama
    LANGCHAIN_OLLAMA = True
except Exception:
    LANGCHAIN_OLLAMA = False

# Try to use official ollama python client
try:
    import ollama
    OLLAMA_PY = True
except Exception:
    OLLAMA_PY = False

# Try HuggingFace transformers pipeline
try:
    from transformers import pipeline
    HF_PIPELINE = True
except Exception:
    HF_PIPELINE = False

from .config import (
    FAQ_PATH, FAQ_SIM_THRESHOLD, RETRIEVAL_TOP_K, SYSTEM_PROMPT,
    OLLAMA_MODEL, DATA_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL, QA_MODEL
)
from .embeddings import get_local_embeddings, embed_texts, get_embeddings_model
from .retriever import retrieve_docs, build_or_load_chroma, create_or_load_db
from .data_loader import load_pdfs_as_page_docs, load_and_split_docs
from .website_loader import fetch_fast_content


# =============================================================================
# Utility Functions
# =============================================================================

def load_faqs(path: str) -> List[dict]:
    """Load FAQ data from JSON file."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _l2_normalize(v):
    """L2 normalize a vector."""
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / (norm + 1e-12)


def _split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Simple character-based text splitter."""
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


# =============================================================================
# LLM Wrappers
# =============================================================================

class OllamaWrapper:
    """Wrapper for Ollama LLM (via langchain or native client)."""
    
    def __init__(self, model_name: str = None):
        if LANGCHAIN_OLLAMA:
            self.llm = ChatOllama(model=model_name or OLLAMA_MODEL)
            self.use_langchain = True
        elif OLLAMA_PY:
            self.model = model_name or OLLAMA_MODEL
            self.use_langchain = False
        else:
            raise RuntimeError("Install langchain_ollama or ollama python client to use Ollama locally.")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.use_langchain:
            return self.llm.predict(prompt)
        else:
            try:
                if len(prompt) > 4000:
                    prompt = prompt[:4000] + "\n\n[Context truncated due to length]"
                
                resp = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "num_predict": 300,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "stop": ["\n\nUser:", "\n\nuser:", "User:", "user:"],
                    },
                    stream=False
                )
                
                if isinstance(resp, dict):
                    answer = resp.get("response", "")
                    if not answer or len(answer.strip()) == 0:
                        return "I apologize, but I couldn't generate a proper response. Please try again."
                    if "logprobs=" in answer:
                        answer = answer.split("logprobs=")[0].strip()
                    return answer.strip()
                else:
                    return "I apologize, but I received an unexpected response format."
                    
            except Exception as e:
                print(f"[ERROR] Ollama error: {e}")
                import traceback
                traceback.print_exc()
                return "I apologize, but I encountered an error generating a response."


class HFQAWrapper:
    """Wrapper for HuggingFace question-answering pipeline."""
    
    def __init__(self, model_name: str = None):
        if not HF_PIPELINE:
            raise RuntimeError("Install transformers to use HuggingFace QA pipeline.")
        self.pipeline = pipeline("question-answering", model=model_name or QA_MODEL, device=-1)
    
    def answer(self, question: str, context: str) -> Optional[str]:
        try:
            result = self.pipeline(question=question, context=context)
            return result.get("answer")
        except Exception as e:
            print(f"[ERROR] HF QA error: {e}")
            return None


# =============================================================================
# Global Singleton State (for lightweight initialization)
# =============================================================================

_vectordb = None
_qa_pipeline = None
_embedding_model = None
_questions = []
_answers = []
_question_embeddings = None
_initialized = False


def initialize_rag(force_reload: bool = False):
    """
    Initialize RAG components as singletons:
    - Embedding model
    - Vector DB from PDFs
    - HuggingFace QA pipeline
    - Predefined QA embeddings from QA.json
    """
    global _vectordb, _qa_pipeline, _embedding_model
    global _questions, _answers, _question_embeddings, _initialized

    if not force_reload and _initialized:
        return

    # 1) Embedding model
    print("🔹 Loading embedding model...")
    try:
        _embedding_model = get_embeddings_model(EMBEDDING_MODEL)
    except Exception as e:
        print(f"[initialize_rag] Embedding model failed: {e}")
        _embedding_model = None

    # 2) Vector DB for PDF docs
    try:
        print("🔹 Building/Loading vector DB from PDF/data...")
        splits = load_and_split_docs(DATA_PATH)
        _vectordb = create_or_load_db(splits, _embedding_model, VECTOR_DB_PATH)
    except Exception as e:
        print(f"[initialize_rag] Vector DB load failed: {e}")
        _vectordb = None

    # 3) HuggingFace QA pipeline
    if HF_PIPELINE:
        try:
            print(f"🔹 Loading QA model: {QA_MODEL}")
            _qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=-1)
        except Exception as e:
            print(f"[initialize_rag] QA pipeline load failed: {e}")
            _qa_pipeline = None

    # 4) Load predefined QA.json and precompute embeddings
    _load_predefined_qa()

    _initialized = True
    print("✅ RAG initialization complete.\n")


def _load_predefined_qa():
    """Load QA.json and compute question embeddings."""
    global _questions, _answers, _question_embeddings, _embedding_model

    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(base_dir, "docs", "QA.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(base_dir, "..", "docs", "QA.json")
            json_path = os.path.abspath(json_path)

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f)
            _questions = [p.get("question", "").strip() for p in qa_pairs]
            _answers = [p.get("answer", "").strip() for p in qa_pairs]

            if _embedding_model is not None and len(_questions) > 0:
                print("🔹 Embedding predefined QA questions...")
                raw = _embedding_model.embed_documents(_questions)
                arr = np.array(raw, dtype=float)
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                _question_embeddings = arr / norms
    except Exception as e:
        print(f"[_load_predefined_qa] Failed: {e}")
        _questions = []
        _answers = []
        _question_embeddings = None


# =============================================================================
# Search Functions
# =============================================================================

def _search_predefined(question: str, threshold: float = 0.6) -> Optional[str]:
    """Check predefined QA.json for a close match using embedding similarity."""
    global _question_embeddings, _questions, _answers, _embedding_model

    if _question_embeddings is None or _embedding_model is None or len(_questions) == 0:
        return None

    try:
        q_emb = np.array(_embedding_model.embed_query(question), dtype=float)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        scores = np.dot(_question_embeddings, q_emb)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        print(f"[predefined] best_score={best_score:.4f}, matched='{_questions[best_idx]}'")

        if best_score >= threshold:
            return _answers[best_idx]
    except Exception as e:
        print(f"[_search_predefined] Failed: {e}")
    return None


def _search_pdf_rag(question: str, k: int = 3) -> Optional[str]:
    """Use vector DB to find top-k chunks and answer using HF QA model."""
    global _vectordb, _qa_pipeline

    if _vectordb is None or _qa_pipeline is None:
        return None

    try:
        docs = _vectordb.similarity_search(question, k=k)
        if not docs:
            return None
        context = " ".join([doc.page_content for doc in docs])
        result = _qa_pipeline(question=question, context=context)
        return result.get("answer")
    except Exception as e:
        print(f"[_search_pdf_rag] Failed: {e}")
        return None


def _search_website(question: str, k: int = 3) -> Optional[str]:
    """Fetch website content, chunk it, find relevant chunks, run QA."""
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

        chunk_vecs = _embedding_model.embed_documents(chunks)
        arr = np.array(chunk_vecs, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr_norm = arr / norms

        q_emb = np.array(_embedding_model.embed_query(question), dtype=float)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

        scores = np.dot(arr_norm, q_emb)
        top_idx = np.argsort(scores)[-k:][::-1]
        top_chunks = [chunks[i] for i in top_idx if scores[i] > 0.0]

        if not top_chunks:
            return None

        context = " ".join(top_chunks)
        result = _qa_pipeline(question=question, context=context)
        return result.get("answer")
    except Exception as e:
        print(f"[_search_website] Failed: {e}")
        return None


# =============================================================================
# Simple Ask Function (uses global singletons)
# =============================================================================

def ask_question(query: str, predefined_threshold: float = 0.65) -> str:
    """
    Answer a query using fallback chain:
    1) Predefined QA.json
    2) PDF RAG (vectordb + HF QA)
    3) Website scraping + QA
    """
    initialize_rag()

    # 1) Predefined QA
    ans = _search_predefined(query, threshold=predefined_threshold)
    if ans:
        return ans

    # 2) PDF RAG
    ans = _search_pdf_rag(query, k=3)
    if ans:
        return ans

    # 3) Website fallback
    ans = _search_website(query, k=3)
    if ans:
        return ans

    return "Sorry, I couldn't find an authoritative answer. Please check the FAST website or ask an admin."


# =============================================================================
# RAGChat Class (uses Ollama LLM with conversation history)
# =============================================================================

class RAGChat:
    """
    Full-featured RAG chatbot with:
    - FAQ matching
    - ChromaDB retrieval
    - Ollama LLM generation
    - Conversation history support
    """
    
    def __init__(self, embeddings_model: str, data_dir: str, chroma_dir: str):
        # Embeddings
        self.embeddings = get_local_embeddings(model_name=embeddings_model)
        
        # Load FAQs and compute embeddings
        self.faqs = load_faqs(str(FAQ_PATH))
        self.faq_texts = [f.get("question", "") for f in self.faqs]
        if self.faq_texts:
            self.faq_vectors = embed_texts(self.embeddings, self.faq_texts)
        else:
            self.faq_vectors = np.zeros((0, 1))
        
        # Load PDF docs and build vector DB
        print(f"[DEBUG] Loading PDFs from: {data_dir}")
        docs = load_pdfs_as_page_docs(data_dir)
        print(f"[DEBUG] Loaded {len(docs)} document chunks")
        self.db = build_or_load_chroma(docs, chroma_dir, self.embeddings)
        
        # LLM
        self.llm = OllamaWrapper()

    def check_faq(self, user_query: str) -> Tuple[bool, str, float]:
        """Check if query matches an FAQ. Returns (matched, answer, score)."""
        if len(self.faq_texts) == 0:
            return False, "", 0.0
        
        q_vec = np.array(self.embeddings.embed_query(user_query)).reshape(1, -1)
        sims = cosine_similarity(q_vec, self.faq_vectors).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        
        if best_score >= FAQ_SIM_THRESHOLD:
            return True, self.faqs[best_idx]["answer"], best_score
        return False, "", best_score

    def rag_answer(self, user_query: str, conversation_history: List[dict]) -> str:
        """
        Run retrieval, build prompt with conversation history,
        call Ollama LLM, and append sources.
        """
        print(f"[DEBUG] Query: {user_query}")
        docs_and_scores = retrieve_docs(self.db, user_query, k=RETRIEVAL_TOP_K)
        print(f"[DEBUG] Retrieved {len(docs_and_scores)} documents")
        
        if not docs_and_scores:
            return "I don't know — this information isn't available in the provided documents."

        # Build context and collect sources
        context_blocks = []
        sources = set()
        for i, (doc, score) in enumerate(docs_and_scores):
            meta = doc.metadata or {}
            source = meta.get("source", "unknown.pdf")
            page = meta.get("page", "unknown")
            sources.add((source, page))
            print(f"[DEBUG] Doc {i+1}: {source} page {page}, score: {score:.3f}")
            context_blocks.append(f"--- {source} (page {page}) ---\n{doc.page_content}")

        context_text = "\n\n".join(context_blocks)

        # Build history text (last 3 turns only)
        history_text = ""
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        for turn in recent_history:
            role = turn.get("role")
            text = turn.get("text")
            history_text += f"{role}: {text}\n"

        # Compose prompt
        prompt = f"""{SYSTEM_PROMPT}

Context from documents:
{context_text}

Recent conversation:
{history_text}

Current question: {user_query}

Instructions: Answer the question using ONLY the information in the context above. Be concise and direct. If the answer is not in the context, say "I don't have information about that in the provided documents." Do not make up information."""

        resp = self.llm.generate(prompt)
        resp = resp.strip()
        
        # Add sources
        if resp and "Sources:" not in resp and sources:
            sources_line = "\n\nSources: " + ", ".join(
                [f"{s[0]} — page {s[1]}" for s in sorted(sources)]
            )
            resp = f"{resp}{sources_line}"
        
        return resp

    def answer(self, user_query: str, conversation_history: List[dict] = None) -> str:
        """
        Main entry point: check FAQ first, then fall back to RAG.
        """
        if conversation_history is None:
            conversation_history = []
        
        # Check FAQ first
        matched, faq_answer, score = self.check_faq(user_query)
        if matched:
            print(f"[DEBUG] FAQ match with score {score:.3f}")
            return faq_answer
        
        # Fall back to RAG
        return self.rag_answer(user_query, conversation_history)