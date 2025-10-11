# backend/models.py
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email field must be set")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    ROLE_CHOICES = (
        ('candidate', 'Candidate'),
        ('admin', 'Admin'),
    )

    email = models.EmailField(unique=True)
    name = models.CharField(max_length=255)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="candidate")
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)  # Required for admin access


    profile_image = models.ImageField(
        upload_to='profile_pics/', 
        default='profile_pics/default.png', 
        blank=True
    )
    
    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["name"]

    def __str__(self):
        return f"{self.email} ({self.role})"


# Other models
class Expression(models.Model):
    name = models.CharField(max_length=100, unique=True)
    percentage = models.FloatField()

    def __str__(self):
        return f"{self.name} ({self.percentage}%)"


class Eyes(models.Model):
    side = models.CharField(max_length=50)
    percentage = models.FloatField()

    def __str__(self):
        return f"{self.side} Eye ({self.percentage}%)"


class HandsExpression(models.Model):
    move = models.CharField(max_length=100, unique=True)
    percentage = models.FloatField()

    def __str__(self):
        return f"{self.move} ({self.percentage}%)"


class Speech(models.Model):
    word = models.CharField(max_length=255, unique=True)
    percentage = models.FloatField()

    def __str__(self):
        return f"{self.word} ({self.percentage}%)"
