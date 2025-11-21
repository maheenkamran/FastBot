import os, json
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Conversation, Message
from .rag.chain import RAGChat
from .rag.config import EMBEDDING_MODEL, DATA_DIR, CHROMA_PERSIST_DIR, FAQ_PATH

# Global variables
model = None
question_embeddings = None
questions = []
answers = []

# Initialize RAGChat singleton on module import (heavy; adjust if you prefer lazy)
RAG = RAGChat(embeddings_model=EMBEDDING_MODEL, data_dir=str(DATA_DIR), chroma_dir=str(CHROMA_PERSIST_DIR))

@csrf_exempt
def chat_endpoint(request):
    if request.method != "POST":
        return JsonResponse({"error":"POST required"}, status=400)
    try:
        payload = json.loads(request.body)
        conv_id = payload.get("conversation_id")
        question = payload.get("question", "").strip()
        if not question:
            return JsonResponse({"error":"question required"}, status=400)

        # get or create conversation
        if conv_id:
            try:
                conv = Conversation.objects.get(id=conv_id)
            except Conversation.DoesNotExist:
                conv = Conversation.objects.create(title="New conversation")
        else:
            conv = Conversation.objects.create(title="New conversation")

        # append user message
        Message.objects.create(conversation=conv, role="user", text=question)

        # Build conversation history for inclusion in prompt (last N messages)
        msgs = conv.messages.all()
        history = [{"role":m.role, "text":m.text} for m in msgs]

        # 1) FAQ-first
        matched, answer, sim = RAG.check_faq(question)
        if matched:
            # Save assistant message and return exact FAQ answer
            Message.objects.create(conversation=conv, role="assistant", text=answer)
            return JsonResponse({
                "conversation_id": conv.id,
                "answer": answer,
                "used_faq": True,
                "faq_similarity": sim,
                "sources": ["FAQ"]
            })

        # 2) fallback to RAG
        rag_answer = RAG.rag_answer(question, history)
        Message.objects.create(conversation=conv, role="assistant", text=rag_answer)

        # Extract sources line if present
        sources = []
        for line in rag_answer.splitlines():
            if line.lower().startswith("sources"):
                sources = [s.strip() for s in line.split(":",1)[1].split(",")]
                break

        return JsonResponse({
            "conversation_id": conv.id,
            "answer": rag_answer,
            "used_faq": False,
            "sources": sources
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def load_model_once():
    """Load model and QA data into memory once"""
    global model, question_embeddings, questions, answers
    if model is not None:
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

def preload(request):
    load_model_once()
    return HttpResponse("✅ Model preloaded successfully!")

@csrf_exempt
def ask(request):
    global model, question_embeddings, questions, answers

    if model is None:
        return JsonResponse({"error": "Model not loaded yet. Visit /preload/ first."}, status=503)

    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        query = data.get("question", "").strip()
        if not query:
            return JsonResponse({"error": "No question provided"}, status=400)

        query_embedding = model.encode([query], convert_to_tensor=False, normalize_embeddings=True)[0]
        scores = np.dot(question_embeddings, query_embedding)
        best_idx = int(np.argmax(scores))
        
        print("Query received:", query)
        print("Best match:", questions[best_idx], "| Score:", scores[best_idx])

        response = {
            "answer": answers[best_idx],
            "matched_question": questions[best_idx],
            "score": float(scores[best_idx])
        }
        return JsonResponse(response)

    return JsonResponse({"error": "Only POST allowed"}, status=405)

def index(request):
    return HttpResponse("Hello from backend!")