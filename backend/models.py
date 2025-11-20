from django.db import models
from django.contrib.auth.hashers import make_password, check_password
import json

# Create your models here.
class Expression(models.Model):
    name = models.CharField(max_length=100, unique=True)
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.name} ({self.percentage}%)"

class Eyes(models.Model):
    side = models.CharField(max_length=50)  
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.side} Eye ({self.percentage}%)"

class HandsExpression(models.Model):
    move = models.CharField(max_length=100, unique=True)
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.move} ({self.percentage}%)"

class Speech(models.Model):
    word = models.CharField(max_length=255, unique=True)
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.word} ({self.percentage}%)"
    
class User(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128) # Stores hashed password
    role = models.CharField(max_length=20, default='candidate') # 'candidate' or 'admin'

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    def __str__(self):
        return self.email
    
class AnalysisResult(models.Model):
    """Stores the complete analysis result for a user's video."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date_time = models.DateTimeField(auto_now_add=True)
    video_path = models.CharField(max_length=500)

    # 🚨 NEW FIELDS FOR INDIVIDUAL SCORES 🚨
    overall_confidence = models.FloatField()
    expression_confidence = models.FloatField(default=0.0)
    eye_movement_confidence = models.FloatField(default=0.0)
    speech_confidence = models.FloatField(default=0.0)
    hand_gesture_confidence = models.FloatField(default=0.0)
    speech_details = models.TextField(default="[]")

    # This field still stores the full detail (including expression_seconds)
    detailed_results = models.TextField()

    def get_detailed_results(self):
        """Utility to deserialize results from TextField."""
        try:
            return json.loads(self.detailed_results)
        except json.JSONDecodeError:
            return {}

    def get_speech_details(self):
        try:
            return json.loads(self.speech_details)
        except:
            return []

    def __str__(self):
        return f"Analysis for {self.user.email} on {self.date_time.strftime('%Y-%m-%d %H:%M')}"