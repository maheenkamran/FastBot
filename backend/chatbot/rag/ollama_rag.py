import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import os

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

from .config import FAQ_PATH, FAQ_SIM_THRESHOLD, RETRIEVAL_TOP_K, SYSTEM_PROMPT, OLLAMA_MODEL
from .embeddings import get_local_embeddings, embed_texts
from .retriever import retrieve_docs, build_or_load_chroma
from .data_loader import load_pdfs_as_page_docs

def load_faqs(path: str) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

class OllamaWrapper:
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
                # Limit prompt length to avoid memory issues
                if len(prompt) > 4000:
                    prompt = prompt[:4000] + "\n\n[Context truncated due to length]"
                
                print(f"[DEBUG] Sending to Ollama, prompt length: {len(prompt)} chars")
                print(f"[DEBUG] Model: {self.model}")
                
                resp = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "num_predict": 300,  # Max tokens to generate
                        "temperature": 0.3,   # Lower temperature for more focused answers
                        "top_p": 0.9,
                        "stop": ["\n\nUser:", "\n\nuser:", "User:", "user:"],  # Stop at next turn
                    },
                    stream=False  # Explicitly disable streaming
                )
                
                print(f"[DEBUG] Raw response type: {type(resp)}")
                
                if isinstance(resp, dict):
                    # Extract the actual text response
                    answer = resp.get("response", "")
                    
                    # Debug: print first 200 chars
                    print(f"[DEBUG] Response preview: {answer[:200]}")
                    
                    # Clean up any artifacts
                    if not answer or len(answer.strip()) == 0:
                        return "I apologize, but I couldn't generate a proper response. Please try again."
                    
                    # Remove trailing metadata/debug info
                    if "logprobs=" in answer:
                        answer = answer.split("logprobs=")[0].strip()
                    
                    # Clean up lines that look like token IDs or metadata
                    lines = answer.split('\n')
                    clean_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip lines that are mostly numbers/commas (token IDs)
                        if len(line) > 20 and sum(c.isdigit() or c == ',' or c == ' ' for c in line) / len(line) > 0.7:
                            print(f"[DEBUG] Skipping suspicious line: {line[:50]}...")
                            continue
                        # Skip metadata lines
                        if line.startswith('[') or line.startswith('(') or 'logprobs=' in line:
                            continue
                        if line:
                            clean_lines.append(line)
                    
                    answer = '\n'.join(clean_lines).strip()
                    
                    if not answer:
                        return "I couldn't generate a proper response. Please try rephrasing your question."
                    
                    return answer
                else:
                    print(f"[ERROR] Unexpected response type: {type(resp)}")
                    return "I apologize, but I received an unexpected response format."
                    
            except Exception as e:
                print(f"[ERROR] Ollama error: {e}")
                import traceback
                traceback.print_exc()
                return "I apologize, but I encountered an error generating a response. Please try rephrasing your question."

class RAGChat:
    def __init__(self, embeddings_model: str, data_dir: str, chroma_dir: str):
        # embeddings
        self.embeddings = get_local_embeddings(model_name=embeddings_model)
        # load faqs and compute faq embeddings
        self.faqs = load_faqs(str(FAQ_PATH))
        self.faq_texts = [f.get("question","") for f in self.faqs]
        if self.faq_texts:
            self.faq_vectors = embed_texts(self.embeddings, self.faq_texts)
        else:
            self.faq_vectors = np.zeros((0,1))
        # load pdf docs and build vector db
        print(f"[DEBUG] Loading PDFs from: {data_dir}")
        docs = load_pdfs_as_page_docs(data_dir)
        print(f"[DEBUG] Loaded {len(docs)} document chunks")
        self.db = build_or_load_chroma(docs, chroma_dir, self.embeddings)
        # LLM
        self.llm = OllamaWrapper()

    def check_faq(self, user_query: str) -> Tuple[bool, str, float]:
        """
        Return (matched, answer, score)
        """
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
        Run retrieval, build prompt with conversation history and system prompt,
        call LLM, and append Sources line.
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
            print(f"[DEBUG] Doc {i+1}: {source} page {page}, score: {score:.3f}, content length: {len(doc.page_content)}")
            context_blocks.append(f"--- {source} (page {page}) ---\n{doc.page_content}")

        context_text = "\n\n".join(context_blocks)
        print(f"[DEBUG] Total context length: {len(context_text)} chars")

        # Build history text (keep it short - last 3 turns only)
        history_text = ""
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        for turn in recent_history:
            role = turn.get("role")
            text = turn.get("text")
            history_text += f"{role}: {text}\n"

        # Compose a cleaner, more direct prompt
        prompt = f"""{SYSTEM_PROMPT}

Context from documents:
{context_text}

Recent conversation:
{history_text}

Current question: {user_query}

Instructions: Answer the question using ONLY the information in the context above. Be concise and direct. If the answer is not in the context, say "I don't have information about that in the provided documents." Do not make up information."""

        print(f"[DEBUG] Final prompt length: {len(prompt)} chars")
        
        resp = self.llm.generate(prompt)
        resp = resp.strip()
        
        # Add sources if not already present
        if resp and "Sources:" not in resp and sources:
            sources_line = "\n\nSources: " + ", ".join([f"{s[0]} — page {s[1]}" for s in sorted(sources)])
            resp = f"{resp}{sources_line}"
        
        return resp