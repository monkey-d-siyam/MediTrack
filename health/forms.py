"""
Forms for MediTrack health app.
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import HealthLog


class RegisterForm(UserCreationForm):
    """Extended user registration form with email field."""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'your@email.com',
        })
    )
    first_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'First Name',
        })
    )
    last_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Last Name',
        })
    )

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Apply Bootstrap classes to all fields
        for field_name, field in self.fields.items():
            field.widget.attrs.setdefault('class', 'form-control')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data.get('first_name', '')
        user.last_name = self.cleaned_data.get('last_name', '')
        if commit:
            user.save()
        return user


class HealthLogForm(forms.ModelForm):
    """Form for submitting a new health data entry."""

    class Meta:
        model = HealthLog
        fields = ['heart_rate', 'sleep_hours', 'steps', 'bmi', 'notes']
        widgets = {
            'heart_rate': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. 72',
                'min': 30, 'max': 220, 'step': 0.1,
            }),
            'sleep_hours': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. 7.5',
                'min': 0, 'max': 24, 'step': 0.5,
            }),
            'steps': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. 8500',
                'min': 0, 'max': 100000,
            }),
            'bmi': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. 22.5',
                'min': 10, 'max': 60, 'step': 0.1,
            }),
            'notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Any notes about how you felt today? (optional)',
            }),
        }
        labels = {
            'heart_rate': 'Heart Rate (bpm)',
            'sleep_hours': 'Sleep Duration (hours)',
            'steps': 'Daily Steps',
            'bmi': 'BMI',
            'notes': 'Notes (Optional)',
        }
        help_texts = {
            'heart_rate': 'Normal range: 60–100 bpm',
            'sleep_hours': 'Recommended: 7–9 hours',
            'steps': 'Goal: 10,000 steps/day',
            'bmi': 'Healthy range: 18.5–24.9',
        }
