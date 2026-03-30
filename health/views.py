"""
Views for MediTrack health app (Upgraded).

Sections:
  1. Authentication views
  2. Dashboard view
  3. Health data input view
  4. History & detail views
  5. ML Insights view
  6. API / AJAX endpoints
"""

import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Avg

from .models import HealthLog
from .forms import RegisterForm, HealthLogForm
from .services import evaluate_health
from .ml_model import get_model_evaluation, get_feature_importance


# ─── 1. Authentication ────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, f"Welcome to MediTrack, {user.first_name or user.username}! 🎉")
        return redirect('dashboard')
    return render(request, 'health/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            login(request, form.get_user())
            messages.success(request, f"Welcome back, {form.get_user().first_name or form.get_user().username}!")
            return redirect('dashboard')
        messages.error(request, "Invalid username or password.")
    return render(request, 'health/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')


# ─── 2. Dashboard ─────────────────────────────────────────────────────────────

@login_required
def dashboard_view(request):
    logs       = HealthLog.objects.filter(user=request.user)
    latest_log = logs.first()

    # Chart data (last 14 entries in chronological order)
    chart_logs   = list(logs[:14])[::-1]
    chart_labels = [log.created_at.strftime('%b %d') for log in chart_logs]
    chart_scores = [log.score for log in chart_logs]
    chart_hr     = [log.heart_rate for log in chart_logs]
    chart_sleep  = [log.sleep_hours for log in chart_logs]
    chart_steps  = [log.steps for log in chart_logs]

    # Risk distribution for all logs (for the donut chart)
    risk_counts = {
        'low':    logs.filter(risk_level='low').count(),
        'medium': logs.filter(risk_level='medium').count(),
        'high':   logs.filter(risk_level='high').count(),
    }

    stats = logs.aggregate(
        avg_score=Avg('score'),
        avg_hr=Avg('heart_rate'),
        avg_sleep=Avg('sleep_hours'),
        avg_steps=Avg('steps'),
        avg_bmi=Avg('bmi'),
    )

    # Re-generate suggestions + explanations for latest log
    suggestions  = []
    explanations = []
    if latest_log:
        result       = evaluate_health(latest_log.heart_rate, latest_log.sleep_hours,
                                       latest_log.steps, latest_log.bmi)
        suggestions  = result['suggestions']
        explanations = result['explanations']

    context = {
        'latest_log':    latest_log,
        'recent_logs':   logs[:5],
        'total_logs':    logs.count(),
        'stats':         stats,
        'risk_counts':   risk_counts,
        'suggestions':   suggestions,
        'explanations':  explanations,
        'chart_labels':  json.dumps(chart_labels),
        'chart_scores':  json.dumps(chart_scores),
        'chart_hr':      json.dumps(chart_hr),
        'chart_sleep':   json.dumps(chart_sleep),
        'chart_steps':   json.dumps(chart_steps),
        'risk_counts_json': json.dumps([risk_counts['low'], risk_counts['medium'], risk_counts['high']]),
    }
    return render(request, 'health/dashboard.html', context)


# ─── 3. Health Data Input ─────────────────────────────────────────────────────

@login_required
def log_health_view(request):
    form = HealthLogForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        log_entry      = form.save(commit=False)
        log_entry.user = request.user

        result = evaluate_health(
            heart_rate=log_entry.heart_rate,
            sleep_hours=log_entry.sleep_hours,
            steps=log_entry.steps,
            bmi=log_entry.bmi,
        )
        log_entry.score      = result['score']
        log_entry.risk_level = result['risk_level']
        log_entry.save()

        messages.success(
            request,
            f"Health data logged! Score: {result['score']:.0f}/100 · "
            f"Risk: {result['risk_level'].capitalize()} · "
            f"Confidence: {result['confidence']*100:.0f}%"
        )
        return redirect('dashboard')
    return render(request, 'health/log_health.html', {'form': form})


# ─── 4. History & Detail ──────────────────────────────────────────────────────

@login_required
def history_view(request):
    logs = HealthLog.objects.filter(user=request.user)
    return render(request, 'health/history.html', {'logs': logs})


@login_required
def log_detail_view(request, pk):
    log    = get_object_or_404(HealthLog, pk=pk, user=request.user)
    result = evaluate_health(log.heart_rate, log.sleep_hours, log.steps, log.bmi)
    return render(request, 'health/log_detail.html', {
        'log':          log,
        'suggestions':  result['suggestions'],
        'explanations': result['explanations'],
        'probabilities': result.get('probabilities', {}),
        'confidence':   result.get('confidence', 1.0),
    })


@login_required
def delete_log_view(request, pk):
    log = get_object_or_404(HealthLog, pk=pk, user=request.user)
    if request.method == 'POST':
        log.delete()
        messages.success(request, "Health log entry deleted.")
    return redirect('history')


# ─── 5. ML Insights View ─────────────────────────────────────────────────────

@login_required
def ml_insights_view(request):
    """
    Dedicated page showing:
      - Model type + training summary
      - Test accuracy + CV score + macro F1
      - Confusion matrix (rendered via Chart.js)
      - Feature importance bar chart (rendered via Chart.js)
      - Per-class precision / recall / F1
    """
    evaluation = get_model_evaluation()
    importance = get_feature_importance()

    # Prepare confusion matrix as flat lists for Chart.js
    cm = evaluation.get('confusion_matrix', [])
    cm_flat    = [cell for row in cm for cell in row]
    cm_labels  = evaluation.get('class_labels', [])

    # Feature names and importance percentages for the chart
    feat_names = [item['feature']    for item in importance]
    feat_vals  = [item['percentage'] for item in importance]

    context = {
        'evaluation':   evaluation,
        'importance':   importance,
        'cm_flat':      json.dumps(cm_flat),
        'cm_labels':    json.dumps(cm_labels),
        'cm_size':      len(cm_labels),
        'feat_names':   json.dumps(feat_names),
        'feat_vals':    json.dumps(feat_vals),
    }
    return render(request, 'health/ml_insights.html', context)


# ─── 6. API Endpoints ─────────────────────────────────────────────────────────

@login_required
def api_chart_data(request):
    n    = int(request.GET.get('n', 14))
    logs = list(HealthLog.objects.filter(user=request.user)[:n])[::-1]
    return JsonResponse({
        'labels':      [log.created_at.strftime('%b %d') for log in logs],
        'scores':      [log.score for log in logs],
        'heart_rate':  [log.heart_rate for log in logs],
        'sleep_hours': [log.sleep_hours for log in logs],
        'steps':       [log.steps for log in logs],
        'bmi':         [log.bmi for log in logs],
    })


@login_required
def api_predict_preview(request):
    """
    AJAX endpoint for the live score preview on the log form.
    Calls the full ML pipeline and returns score + risk + confidence.
    """
    try:
        hr    = float(request.GET.get('hr', 0))
        sleep = float(request.GET.get('sleep', 0))
        steps = int(request.GET.get('steps', 0))
        bmi   = float(request.GET.get('bmi', 0))
        if not all([hr, sleep, steps, bmi]):
            return JsonResponse({'error': 'Missing fields'}, status=400)
        result = evaluate_health(hr, sleep, steps, bmi)
        return JsonResponse({
            'score':         result['score'],
            'risk_level':    result['risk_level'],
            'confidence':    result['confidence'],
            'probabilities': result['probabilities'],
            'explanations':  result['explanations'],
        })
    except (ValueError, TypeError) as e:
        return JsonResponse({'error': str(e)}, status=400)
