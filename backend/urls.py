from django.contrib import admin
from django.urls import path
from .import views
# urls.py
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('upload_video', views.upload_video, name='upload_video'),
    path('home', views.home, name='home'),
    path('', views.auth, name='auth'),
    path('profile/', views.candidate_profile, name='candidate_profile'),
    path("signup/", views.signup_view, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
