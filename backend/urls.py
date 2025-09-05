from django.contrib import admin
from django.urls import path
from .import views



urlpatterns = [
    path('upload_video', views.upload_video, name='upload_video'),
    path('', views.home, name='home'),
    path('admin-panel/', views.admin_panel, name='admin_panel'),
    path('history/', views.history, name='history'),
    path('report/', views.report, name='report'),
]
