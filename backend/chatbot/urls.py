from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('preload/', views.preload),
    path('chat/', views.chat_endpoint),
    path('ask/', views.ask),
    path('response/', views.chatbot_response),
]