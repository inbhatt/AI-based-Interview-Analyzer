from django.db import models

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

# --- Admin model for email login ---
class AdminManager(BaseUserManager):
    def create_user(self, email, password=None):
        if not email:
            raise ValueError("Admin must have an email address")
        user = self.model(email=self.normalize_email(email))
        user.set_password(password)
        user.save(using=self._db)
        return user

class AdminUser(AbstractBaseUser):
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name']

    objects = AdminManager()

    def __str__(self):
        return self.email


# --- Candidate and Report models ---
class Candidate(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    contact = models.CharField(max_length=15)
    overall_confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class InterviewReport(models.Model):
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name="reports")
    video = models.FileField(upload_to="candidate_videos/")
    confidence_score = models.FloatField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.candidate.name} - {self.confidence_score}%"

