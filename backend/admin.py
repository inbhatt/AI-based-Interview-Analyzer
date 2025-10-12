from django.contrib import admin

# Register your models here.
from .models import (
    Expression,
    Eyes,
    HandsExpression,
    Speech,
    User,
    AnalysisResult
)

# 1. Register Video Processing Models
admin.site.register(Expression)
admin.site.register(Eyes)
admin.site.register(HandsExpression)
admin.site.register(Speech)

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'role')
    search_fields = ('email', 'name')
    list_filter = ('role',)

# 3. Register Analysis Result Model
@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    # 🚨 UPDATED list_display to show all scores 🚨
    list_display = (
        'id', 
        'user', 
        'date_time', 
        'overall_confidence', 
        'expression_confidence', 
        'eye_movement_confidence', 
        'speech_confidence', 
        'hand_gesture_confidence'
    )
    
    list_display_links = ('id', 'user', 'date_time')
    
    list_filter = ('date_time', 'overall_confidence')
    
    search_fields = ('user__email', 'video_path')
    
    # 🚨 UPDATED fields to show all scores in the detail view 🚨
    fields = (
        'user', 
        'overall_confidence',
        'expression_confidence', 
        'eye_movement_confidence', 
        'speech_confidence', 
        'hand_gesture_confidence',
        'video_path', 
        'detailed_results'
    )
    
    readonly_fields = ('date_time',)