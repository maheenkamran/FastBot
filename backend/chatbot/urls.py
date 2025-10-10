from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/ask/', views.ask, name='ask'),
    path('preload/', views.preload, name='preload')   # 👈 new route
]
