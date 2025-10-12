from django.contrib import admin
from django.urls import path
from .import views



urlpatterns = [
    path('upload_video', views.upload_video, name='upload_video'),
    path('', views.home, name='home'),
]
