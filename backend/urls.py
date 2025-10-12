from django.contrib import admin
from django.urls import path
from .import views



urlpatterns = [
    # General & Authentication Paths
    path('upload_video', views.upload_video, name='upload_video'),
    path('auth', views.auth, name='auth'), # Assuming you have an 'auth' view for login/signup
    path('', views.home_redirect, name='home_redirect'), # Added a redirect for the base path
    path('home', views.home, name='home'),
    path('signup', views.signup_user, name='signup'),
    path('login', views.login_user, name='login'),
    path('logout', views.logout_user, name='logout'),
    
    # Candidate-Specific Paths
    path('history', views.candidate_history, name='candidate_history'),
    
    # Admin Paths
    path('admin-dashboard', views.admin_dashboard, name='admin_dashboard'), 
    path('render-add-admin', views.render_add_admin, name='render_add_admin'), # ⬅️ FIX: This path was missing
    path('add-admin-user', views.add_admin_user, name='add_admin_user'),
    path('create-candidate', views.create_candidate_page, name='create_candidate_page'),
    path('candidate/<int:user_id>/performance', views.candidate_performance, name='candidate_performance'),
    path('candidate/<int:user_id>/upload', views.admin_upload_candidate_video, name='admin_upload_candidate_video'),
    #path('candidate/<int:user_id>/record', views.admin_record_candidate_video, name='admin_record_candidate_video'),
    #path('candidate/<int:user_id>/process-live', views.admin_process_live_video, name='admin_process_live_video'),
    
    # Report Generator Paths
    path('reports', views.report_generator, name='report_generator_self'), 
    path('reports/<int:user_id>', views.report_generator, name='report_generator_admin'), 
]
