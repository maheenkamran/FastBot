# backend/chatbot/views.py
import os
import json
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .rag.chain import ask_question, initialize_rag

# ---------------------------
# Optional: preload models & RAG on import
# ---------------------------
# If this takes too long, you can call /preload/ route instead
try:
    initialize_rag()
except Exception as exc:
    print(f"[views] initialize_rag() raised: {exc}")

# ---------------------------
# Original variables from your code (for embedding approach)
# ---------------------------
model = None                    # SentenceTransformer model
question_embeddings = None      # Precomputed embeddings
questions = []                  # List of questions
answers = []                    # Corresponding answers

def load_model_once(force_reload=False):
    """
    Loads the model and QA dataset into memory once, or reload if force_reload=True
    """
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

# ---------------------------
# Routes / Views
# ---------------------------
def index(request):
    return HttpResponse("Hello from backend!")

@csrf_exempt
def preload(request):
    """
    Preloads model & RAG logic.
    """
    try:
        # Initialize RAG
        initialize_rag(force_reload=True)
        # Also load original model QA embeddings
        load_model_once(force_reload=True)
        return HttpResponse("✅ Model & RAG preloaded successfully!")
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

@csrf_exempt
def ask(request):
    """
    POST endpoint expecting JSON: { "question": "..." }
    Supports both RAG approach and original embeddings approach.
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

    # Try RAG first
    try:
        answer_rag = ask_question(query)
    except Exception as exc:
        print(f"[views.ask] RAG failed: {exc}")
        answer_rag = None

    # Fall back to original embedding-based QA
    try:
        load_model_once()
        query_embedding = model.encode([query], convert_to_tensor=False, normalize_embeddings=True)[0]
        scores = np.dot(question_embeddings, query_embedding)
        best_idx = int(np.argmax(scores))
        answer_embed = answers[best_idx]
    except Exception as exc:
        print(f"[views.ask] Embedding fallback failed: {exc}")
        answer_embed = None
        best_idx = None
        scores = [0]

    response = {
        "answer": answer_rag if answer_rag else answer_embed,
        "matched_question": questions[best_idx] if best_idx is not None else None,
        "score": float(scores[best_idx]) if best_idx is not None else None
    }

    return JsonResponse(response)

def chatbot_response(request):
    """
    Legacy GET endpoint for quick testing: ?q=your_question
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
