
from .models import User
from django import forms

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['name', 'email', 'role','profile_image']  # match your User model fields
