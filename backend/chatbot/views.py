# backend/chatbot/views.py
import os
import json
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Conversation, Message
from .rag.chain import RAGChat, ask_question, initialize_rag
from .rag.config import EMBEDDING_MODEL, DATA_DIR, CHROMA_PERSIST_DIR, FAQ_PATH


# =============================================================================
# Global State
# =============================================================================

# RAGChat instance (for conversation-aware chat)
RAG = None

# Original embedding-based QA (SentenceTransformer approach)
model = None
question_embeddings = None
questions = []
answers = []


# =============================================================================
# Initialization Functions
# =============================================================================

def _init_rag_chat():
    """Initialize RAGChat singleton lazily."""
    global RAG
    if RAG is None:
        try:
            print("🔹 Initializing RAGChat...")
            RAG = RAGChat(
                embeddings_model=EMBEDDING_MODEL,
                data_dir=str(DATA_DIR),
                chroma_dir=str(CHROMA_PERSIST_DIR)
            )
            print("✅ RAGChat initialized!")
        except Exception as e:
            print(f"[_init_rag_chat] Failed: {e}")
            RAG = None
    return RAG


def load_model_once(force_reload=False):
    """Load SentenceTransformer model and QA data into memory once."""
    global model, question_embeddings, questions, answers
    
    if model is not None and not force_reload:
        return

    print("⏳ Preloading model and QA data...")
    from sentence_transformers import SentenceTransformer

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "docs", "QA.json"))

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    with open(json_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    questions = [pair["question"] for pair in qa_pairs]
    answers = [pair["answer"] for pair in qa_pairs]

    question_embeddings = np.array(
        model.encode(questions, convert_to_tensor=False, normalize_embeddings=True)
    )
    print("✅ Model preloaded successfully!\n")


# Try to initialize RAG on module import (can be slow; remove if prefer lazy loading)
try:
    initialize_rag()
except Exception as exc:
    print(f"[views] initialize_rag() raised: {exc}")


# =============================================================================
# Views / Endpoints
# =============================================================================

def index(request):
    """Health check endpoint."""
    return HttpResponse("Hello from backend!")


@csrf_exempt
def preload(request):
    """
    Preload all models and RAG components.
    GET /preload/
    """
    try:
        # Initialize simple RAG (global singletons)
        initialize_rag(force_reload=True)
        # Initialize RAGChat instance
        global RAG
        RAG = None  # Force re-init
        _init_rag_chat()
        # Load original SentenceTransformer model
        load_model_once(force_reload=True)
        return HttpResponse("✅ All models & RAG preloaded successfully!")
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@csrf_exempt
def chat_endpoint(request):
    """
    Main chat endpoint with conversation history support.
    POST /chat/
    Body: { "conversation_id": optional, "question": "..." }
    
    Uses RAGChat with FAQ matching and Ollama LLM.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    try:
        payload = json.loads(request.body)
        conv_id = payload.get("conversation_id")
        question = payload.get("question", "").strip()
        
        if not question:
            return JsonResponse({"error": "question required"}, status=400)

        # Ensure RAGChat is initialized
        rag = _init_rag_chat()
        if rag is None:
            return JsonResponse({"error": "RAG system not initialized"}, status=503)

        # Get or create conversation
        if conv_id:
            try:
                conv = Conversation.objects.get(id=conv_id)
            except Conversation.DoesNotExist:
                conv = Conversation.objects.create(title="New conversation")
        else:
            conv = Conversation.objects.create(title="New conversation")

        # Append user message
        Message.objects.create(conversation=conv, role="user", text=question)

        # Build conversation history
        msgs = conv.messages.all()
        history = [{"role": m.role, "text": m.text} for m in msgs]

        # 1) Try FAQ first
        matched, answer, sim = rag.check_faq(question)
        if matched:
            Message.objects.create(conversation=conv, role="assistant", text=answer)
            return JsonResponse({
                "conversation_id": conv.id,
                "answer": answer,
                "used_faq": True,
                "faq_similarity": sim,
                "sources": ["FAQ"]
            })

        # 2) Fall back to RAG with Ollama
        rag_answer = rag.rag_answer(question, history)
        Message.objects.create(conversation=conv, role="assistant", text=rag_answer)

        # Extract sources line if present
        sources = []
        for line in rag_answer.splitlines():
            if line.lower().startswith("sources"):
                sources = [s.strip() for s in line.split(":", 1)[1].split(",")]
                break

        return JsonResponse({
            "conversation_id": conv.id,
            "answer": rag_answer,
            "used_faq": False,
            "sources": sources
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def ask(request):
    """
    Simple ask endpoint (no conversation history).
    POST /ask/
    Body: { "question": "..." }
    
    Tries RAG first, then falls back to embedding-based QA.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    query = data.get("question", "").strip()
    if not query:
        return JsonResponse({"error": "No question provided"}, status=400)

    # Try RAG first (uses global singletons from chain.py)
    answer_rag = None
    try:
        answer_rag = ask_question(query)
    except Exception as exc:
        print(f"[views.ask] RAG failed: {exc}")

    # Fall back to original embedding-based QA
    answer_embed = None
    best_idx = None
    score = None
    try:
        load_model_once()
        if model is not None:
            query_embedding = model.encode(
                [query], convert_to_tensor=False, normalize_embeddings=True
            )[0]
            scores = np.dot(question_embeddings, query_embedding)
            best_idx = int(np.argmax(scores))
            score = float(scores[best_idx])
            answer_embed = answers[best_idx]
            
            print(f"Query: {query}")
            print(f"Best match: {questions[best_idx]} | Score: {score:.4f}")
    except Exception as exc:
        print(f"[views.ask] Embedding fallback failed: {exc}")

    # Determine final answer
    final_answer = answer_rag if answer_rag else answer_embed
    if not final_answer:
        final_answer = "Sorry, I couldn't find an answer to your question."

    response = {
        "answer": final_answer,
        "matched_question": questions[best_idx] if best_idx is not None else None,
        "score": score,
        "used_rag": answer_rag is not None
    }

    return JsonResponse(response)


def chatbot_response(request):
    """
    Legacy GET endpoint for quick testing.
    GET /response/?q=your_question
    """
    query = request.GET.get("q")
    if not query:
        return JsonResponse({"error": "No question provided"}, status=400)

    try:
        answer = ask_question(query)
        return JsonResponse({"answer": answer})
    except Exception as exc:
        print(f"[views.chatbot_response] error: {exc}")
        return JsonResponse({"error": "Internal server error"}, status=500)