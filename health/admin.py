"""
Django admin configuration for the health app.
Provides a rich admin interface for managing health logs.
"""

from django.contrib import admin
from .models import HealthLog


@admin.register(HealthLog)
class HealthLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'score', 'risk_level', 'heart_rate', 'sleep_hours', 'steps', 'bmi', 'created_at']
    list_filter = ['risk_level', 'created_at', 'user']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['score', 'risk_level', 'created_at']
    ordering = ['-created_at']

    fieldsets = (
        ('User', {'fields': ('user',)}),
        ('Health Metrics', {'fields': ('heart_rate', 'sleep_hours', 'steps', 'bmi')}),
        ('AI Results', {'fields': ('score', 'risk_level')}),
        ('Meta', {'fields': ('notes', 'created_at')}),
    )
