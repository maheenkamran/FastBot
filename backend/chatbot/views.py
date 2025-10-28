import os,json #Imports os for file path operations, Imports json to read/write JSON files (like your QA dataset)
import numpy as np #for numerical operations, like vector embeddings and similarity calculations
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .rag.chain import ask_question

# Global variables
model = None                    #the sentence transformer model
question_embeddings = None      #precomputed embeddings for all questions
questions = []                  #list of questions from QA dataset
answers = []                    #corresponding answers

# A SentenceTransformer is a pretrained model from the sentence-transformers library.
#It converts sentences or paragraphs into dense numerical vectors (embeddings).

def load_model_once():
    #Load model and QA data into memory once 
    
    global model, question_embeddings, questions, answers
    if model is not None: #If the model is already loaded, do nothing (prevents reloading)
        return

    print("⏳ Preloading model and QA data...")
    from sentence_transformers import SentenceTransformer #Imports the SentenceTransformer model class (used for embeddings)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #file:the current Python file where this line is written.,s.path.abspath(__file__) → full absolute path to that file. 
    # #os.path.dirname(...) → the directory containing that file. which is backend folder here
    
    json_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "docs", "QA.json"))
    #Goes two folders up(..,..) and then into docs

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    #Loads the pre-trained sentence transformer model for embeddings.

    #Opens QA.json 
    with open(json_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f) #loads it into list of dictionaries (qa_pairs)
        
    #Extracts questions and answers into separate lists
    questions = [pair["question"] for pair in qa_pairs]
    answers = [pair["answer"] for pair in qa_pairs]
    
    question_embeddings = np.array( #np.array makes 2d array (array of vectors)
        model.encode(questions, convert_to_tensor=False, normalize_embeddings=True)
    )
    #Converts all questions into embeddings using the model.
    #normalize_embeddings=True ensures cosine similarity can be computed using dot product.
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

        query = data.get("question", "").strip() #frontend post request body having question field, we are getting its value
        #question: "bscs requirement?", here question is key and "bscs requirement?" is value, we gwt value of question
        if not query:
            return JsonResponse({"error": "No question provided"}, status=400)

        query_embedding = model.encode([query], convert_to_tensor=False, normalize_embeddings=True)[0]#Converts user query into an embedding.
        scores = np.dot(question_embeddings, query_embedding) #Computes dot product similarity with all preloaded question embeddings.(scores is array)
        best_idx = int(np.argmax(scores)) #Finds the index of the best match (best_idx), which has highest cosine similarity (dot product)
        
        print("Query received:", query)
        print("Best match:", questions[best_idx], "| Score:", scores[best_idx])

        response = {
            "answer": answers[best_idx],
            "matched_question": questions[best_idx],
            "score": float(scores[best_idx])
        }
        return JsonResponse(response)

    return JsonResponse({"error": "Only POST allowed"}, status=405)

def chatbot_response(request):
    query = request.GET.get("q")
    if query:
        answer = ask_question(query)
        return JsonResponse({"answer": answer})
    return JsonResponse({"error": "No question provided"}, status=400)

def index(request):
    return HttpResponse("Hello from backend!")

