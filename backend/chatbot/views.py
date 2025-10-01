from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def ask(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("query", "")
        # For now just echo back
        return JsonResponse({"answer": f"You asked: {query}"})
    return JsonResponse({"error": "Only POST allowed"}, status=405)
