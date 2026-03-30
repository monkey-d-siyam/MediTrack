"""
URL configuration for the health app (Upgraded).
"""

from django.urls import path
from . import views

urlpatterns = [
    # ── Root ──────────────────────────────────────────────────────────────────
    path('', views.dashboard_view, name='home'),

    # ── Auth ──────────────────────────────────────────────────────────────────
    path('register/', views.register_view, name='register'),
    path('login/',    views.login_view,    name='login'),
    path('logout/',   views.logout_view,   name='logout'),

    # ── Core Pages ─────────────────────────────────────────────────────────────
    path('dashboard/',              views.dashboard_view,  name='dashboard'),
    path('log/',                    views.log_health_view, name='log_health'),
    path('history/',                views.history_view,    name='history'),
    path('history/<int:pk>/',       views.log_detail_view, name='log_detail'),
    path('history/<int:pk>/delete/',views.delete_log_view, name='delete_log'),

    # ── ML Insights ───────────────────────────────────────────────────────────
    path('ml-insights/', views.ml_insights_view, name='ml_insights'),

    # ── API ────────────────────────────────────────────────────────────────────
    path('api/chart-data/',      views.api_chart_data,      name='api_chart_data'),
    path('api/predict-preview/', views.api_predict_preview, name='api_predict_preview'),
]
