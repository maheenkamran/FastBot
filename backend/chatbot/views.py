import os,json
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Global variables
model = None
question_embeddings = None
questions = []
answers = []

def load_model_once():
    """Load model and QA data into memory once when Django starts."""
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
#for now we have done preload thing, afterwards we'll do loading as server starts

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
