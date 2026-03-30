"""
Health scoring service.

This module is the single source of truth for:
  - Orchestrating health evaluation (ML + Rules)
  - Rule-based health score computation (0-100)
  - Health tip generation
"""

from .ml_model import predict_risk_ml, explain_prediction

# ─── Configuration ────────────────────────────────────────────────────────────
# True = use ML for risk prediction; False = use rule-based fallback only
USE_ML_MODEL = True

# ─── Rule-Based Health Score ──────────────────────────────────────────────────

def compute_health_score(
    heart_rate: float, sleep_hours: float, steps: int, bmi: float
) -> float:
    """
    Compute a 0–100 health score using weighted rule-based thresholds.
    This score is stored alongside the ML risk prediction as a
    human-readable indicator.
    """
    score = 0.0

    # Heart rate (25 pts)
    if 60 <= heart_rate <= 100:    score += 25.0
    elif 50 <= heart_rate <= 110:  score += 15.0
    else:                           score += 5.0

    # Sleep (25 pts)
    if 7 <= sleep_hours <= 9:              score += 25.0
    elif 6 <= sleep_hours <= 10:           score += 15.0
    elif 5 <= sleep_hours < 6:             score += 8.0
    else:                                   score += 2.0

    # Steps (25 pts)
    if   steps >= 10000: score += 25.0
    elif steps >= 7500:  score += 18.0
    elif steps >= 5000:  score += 12.0
    elif steps >= 2500:  score += 6.0
    else:                score += 1.0

    # BMI (25 pts)
    if   18.5 <= bmi <= 24.9:  score += 25.0
    elif 25.0 <= bmi <= 27.5 or 17.0 <= bmi < 18.5: score += 15.0
    elif 27.5 < bmi <= 30.0:   score += 8.0
    else:                       score += 2.0

    return round(score, 1)


def classify_risk(score: float) -> str:
    """Fallback rule-based risk label from health score."""
    if score >= 70: return 'low'
    if score >= 40: return 'medium'
    return 'high'


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def evaluate_health(
    heart_rate: float, sleep_hours: float, steps: int, bmi: float
) -> dict:
    """
    Single entry point for health evaluation.

    Always computes a rule-based score (stored in DB).
    Risk level comes from ML model (if available) or falls back to rules.

    Returns:
        {
            'score':         float,    # 0–100 health score
            'risk_level':    str,      # 'low' | 'medium' | 'high'
            'probabilities': dict,     # ML class probabilities
            'confidence':    float,    # ML confidence in prediction
            'explanations':  list,     # per-feature status
            'suggestions':   list[str],
        }
    """
    score = compute_health_score(heart_rate, sleep_hours, steps, bmi)

    if USE_ML_MODEL:
        ml_result = predict_risk_ml(heart_rate, sleep_hours, steps, bmi)
        if ml_result:
            risk_level    = ml_result['risk_level']
            probabilities = ml_result['probabilities']
            confidence    = ml_result['confidence']
        else:
            # Fallback if model not trained
            risk_level    = classify_risk(score)
            probabilities = {risk_level: 1.0}
            confidence    = 1.0
    else:
        risk_level    = classify_risk(score)
        probabilities = {}
        confidence    = 1.0

    explanations = explain_prediction(heart_rate, sleep_hours, steps, bmi)
    suggestions  = generate_suggestions(heart_rate, sleep_hours, steps, bmi, score)

    return {
        'score':         score,
        'risk_level':    risk_level,
        'probabilities': probabilities,
        'confidence':    confidence,
        'explanations':  explanations,
        'suggestions':   suggestions,
    }


# ─── Health Tips Generator ─────────────────────────────────────────────────────

def generate_suggestions(
    heart_rate: float, sleep_hours: float, steps: int, bmi: float, score: float
) -> list:
    """Personalised health tips based on current metric values."""
    tips = []

    if heart_rate > 100:
        tips.append("🫀 Elevated resting heart rate detected. Practice diaphragmatic breathing, limit caffeine, and consider a stress audit.")
    elif heart_rate < 60:
        tips.append("🫀 Low resting heart rate. Unless you're a trained athlete, discuss this with a cardiologist.")
    else:
        tips.append("✅ Heart rate is in the ideal resting zone (60–100 bpm). Keep up your cardio routine.")

    if sleep_hours < 5:
        tips.append("😴 Critical sleep deficit. Chronic short sleep is linked to hypertension, weight gain, and cognitive decline. Prioritise sleep immediately.")
    elif sleep_hours < 7:
        tips.append("😴 Slightly under the recommended 7–9 hours. Set a consistent bedtime, avoid bright screens 1 hr before bed.")
    elif sleep_hours > 9:
        tips.append("😴 Oversleeping may indicate poor sleep quality or depression. Track sleep cycles and consider a sleep study.")
    else:
        tips.append("✅ Sleep is in the optimal range. Maintain your consistent schedule for best cognitive performance.")

    if steps < 3000:
        tips.append("🏃 Step count is critically low. Aim to add 1,000 steps every few days — even short walks significantly improve metabolic health.")
    elif steps < 7500:
        tips.append("🚶 Moderate activity level. Targeting 8,000–10,000 daily steps reduces cardiovascular risk by up to 40%.")
    elif steps < 10000:
        tips.append("🚶 Almost at the 10,000-step mark. A 15-minute evening walk closes the gap.")
    else:
        tips.append("✅ Excellent activity level. High step count is strongly correlated with longevity and cardiovascular health.")

    if bmi < 18.5:
        tips.append("⚖️ BMI is below the healthy range. A nutritionist can help create a calorie-dense, nutrient-rich meal plan.")
    elif bmi <= 24.9:
        tips.append("✅ BMI is in the healthy range (18.5–24.9). Sustain with balanced nutrition and regular movement.")
    elif bmi <= 29.9:
        tips.append("⚖️ BMI is in the overweight range. A deficit of 300–500 kcal/day paired with 150 min/week of moderate exercise is evidence-based.")
    else:
        tips.append("⚖️ BMI falls in the obese range. Structured medical supervision often yields the best long-term outcomes.")

    if score >= 80:
        tips.append("🌟 Exceptional overall health score. You're in the top tier — share your habits with others!")
    elif score >= 60:
        tips.append("👍 Good health score. Targeting 1–2 weak metrics could push you to excellent.")
    elif score >= 40:
        tips.append("⚠️ Moderate risk. Small consistent improvements compound into significant health gains over time.")
    else:
        tips.append("🚨 Multiple metrics need attention. Consider a comprehensive health check-up and structured lifestyle intervention.")

    return tips
