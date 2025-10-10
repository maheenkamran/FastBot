from django.apps import AppConfig

class ChatbotConfig(AppConfig):
    name = 'chatbot'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        from .views import load_model_once
        load_model_once()
