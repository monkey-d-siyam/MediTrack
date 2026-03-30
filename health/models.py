"""
Health app models — HealthLog stores all user health data entries.
"""

from django.db import models
from django.contrib.auth.models import User


class HealthLog(models.Model):
    """
    Stores a single health data entry for a user.
    Each log captures vital metrics, a computed health score,
    and an AI-determined risk level.
    """

    RISK_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='health_logs'
    )
    heart_rate = models.FloatField(
        help_text="Heart rate in beats per minute (bpm)"
    )
    sleep_hours = models.FloatField(
        help_text="Hours of sleep last night"
    )
    steps = models.IntegerField(
        help_text="Total steps walked today"
    )
    bmi = models.FloatField(
        help_text="Body Mass Index"
    )
    score = models.FloatField(
        default=0.0,
        help_text="Computed health score (0–100)"
    )
    risk_level = models.CharField(
        max_length=10,
        choices=RISK_CHOICES,
        default='low',
        help_text="Predicted risk level: low / medium / high"
    )
    notes = models.TextField(
        blank=True,
        null=True,
        help_text="Optional user notes for this entry"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Health Log'
        verbose_name_plural = 'Health Logs'

    def __str__(self):
        return f"{self.user.username} — Score: {self.score:.1f} [{self.risk_level}] @ {self.created_at.strftime('%Y-%m-%d')}"

    @property
    def risk_color(self):
        """Returns a CSS color class based on risk level."""
        return {
            'low': 'success',
            'medium': 'warning',
            'high': 'danger',
        }.get(self.risk_level, 'secondary')
