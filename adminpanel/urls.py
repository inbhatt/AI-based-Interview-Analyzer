from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.admin_login, name='admin_login'),
    path('logout/', views.admin_logout, name='admin_logout'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('candidates/', views.candidate_list, name='candidate_list'),
    path('candidate/<int:candidate_id>/', views.candidate_detail, name='candidate_detail'),
    path('add_candidate/', views.add_candidate, name='add_candidate'),
    path('live_record/', views.live_record, name='live_record'),
]
