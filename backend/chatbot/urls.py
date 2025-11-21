from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('preload/', views.preload, name='preload'),
    path('ask/', views.ask, name='ask'),
    path('chat/', views.chat_endpoint, name='chat_endpoint'),  # Changed from chatbot_response
]