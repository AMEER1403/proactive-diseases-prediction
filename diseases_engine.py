"""
disease_engine.py
─────────────────────────────────────────────────────────────────────────────
Proactive Disease Prediction Engine

Maps combinations of lab parameters + daily activity metrics to specific
disease risk probabilities using a rule-weighted scoring system layered on
top of clinical evidence thresholds.

Diseases modelled:
  1.  Type 2 Diabetes
  2.  Hypertension
  3.  Coronary Artery Disease (CAD)
  4.  Anaemia
  5.  Chronic Kidney Disease (CKD)
  6.  Liver Disease (Non-alcoholic)
  7.  Thyroid Disorder (Hypo/Hyper)
  8.  Sleep Apnea / Sleep Disorder
  9.  Metabolic Syndrome
  10. Obesity-related Complications

Each disease entry returns:
  {
    "disease":      str,
    "risk_pct":     float (0–100),
    "risk_tier":    "Low" | "Moderate" | "High" | "Critical",
    "key_drivers":  [str, ...],   # which values triggered the score
    "advice":       str,
  }
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiseaseResult:
    disease:     str
    risk_pct:    float
    risk_tier:   str
    key_drivers: list[str]
    advice:      str
    icon:        str = "🔬"


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _tier(pct: float) -> str:
    if pct < 25:  return "Low"
    if pct < 50:  return "Moderate"
    if pct < 75:  return "High"
    return "Critical"


def _cap(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _get(d: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
#  INDIVIDUAL DISEASE SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def _diabetes(lab: dict, wear: dict) -> DiseaseResult:
    score   = 0.0
    drivers = []

    hba1c   = _get(lab, "hba1c")
    fbs     = _get(lab, "fasting_glucose")
    bmi_prx = _get(wear, "steps")        # low steps → sedentary lifestyle proxy
    stress  = _get(wear, "stress_index")
    sleep   = _get(wear, "sleep_hours")

    # HbA1c — strongest predictor
    if hba1c is not None:
        if hba1c >= 6.5:   score += 40; drivers.append(f"HbA1c {hba1c}% ≥ 6.5 (diabetic range)")
        elif hba1c >= 5.7: score += 22; drivers.append(f"HbA1c {hba1c}% in pre-diabetic range")

    # Fasting glucose
    if fbs is not None:
        if fbs >= 126:    score += 30; drivers.append(f"Fasting glucose {fbs} mg/dL ≥ 126")
        elif fbs >= 100:  score += 15; drivers.append(f"Fasting glucose {fbs} mg/dL (pre-diabetic)")

    # Lifestyle modifiers
    if bmi_prx is not None and bmi_prx < 4000:
        score += 10; drivers.append(f"Very low daily steps ({int(bmi_prx)}) — sedentary risk")
    if stress is not None and stress > 65:
        score += 8;  drivers.append(f"High stress index ({stress}) — cortisol-driven glucose rise")
    if sleep is not None and sleep < 5.5:
        score += 7;  drivers.append(f"Sleep deprivation ({sleep}h) — insulin resistance link")

    advice = (
        "Monitor HbA1c and fasting glucose every 3 months. Follow a low-glycaemic diet, "
        "aim for 7,000+ steps/day, and consult an endocrinologist if HbA1c ≥ 6.5%."
    )
    return DiseaseResult("Type 2 Diabetes", _cap(score), _tier(_cap(score)), drivers, advice, "🩸")


def _hypertension(lab: dict, wear: dict) -> DiseaseResult:
    score   = 0.0
    drivers = []

    sbp    = _get(wear, "systolic_bp") or _get(lab, "systolic_bp")
    dbp    = _get(wear, "diastolic_bp") or _get(lab, "diastolic_bp")
    hr     = _get(wear, "heart_rate")
    stress = _get(wear, "stress_index")
    sleep  = _get(wear, "sleep_hours")
    ldl    = _get(lab, "ldl")
    trigs  = _get(lab, "triglycerides")

    if sbp is not None:
        if sbp >= 180:   score += 50; drivers.append(f"Systolic BP {sbp} mmHg — hypertensive crisis")
        elif sbp >= 140: score += 35; drivers.append(f"Systolic BP {sbp} mmHg — Stage 2 hypertension")
        elif sbp >= 130: score += 20; drivers.append(f"Systolic BP {sbp} mmHg — Stage 1 hypertension")

    if dbp is not None:
        if dbp >= 110:   score += 20; drivers.append(f"Diastolic BP {dbp} mmHg — severely elevated")
        elif dbp >= 90:  score += 12; drivers.append(f"Diastolic BP {dbp} mmHg — elevated")

    if hr is not None and hr > 100:
        score += 8; drivers.append(f"Tachycardia HR {hr} bpm")
    if stress is not None and stress > 70:
        score += 10; drivers.append(f"High stress index ({stress}) — sympathetic activation")
    if sleep is not None and sleep < 6:
        score += 7;  drivers.append(f"Poor sleep ({sleep}h) — nocturnal BP dysregulation")
    if ldl is not None and ldl > 160:
        score += 5;  drivers.append(f"Elevated LDL {ldl} mg/dL — vascular load")
    if trigs is not None and trigs > 200:
        score += 5;  drivers.append(f"High triglycerides {trigs} mg/dL")

    advice = (
        "Track BP twice daily. Limit sodium intake to <2g/day, reduce alcohol, and exercise "
        "aerobically 30 min/day. If BP consistently ≥140/90, seek physician review."
    )
    return DiseaseResult("Hypertension", _cap(score), _tier(_cap(score)), drivers, advice, "🩺")


def _cad(lab: dict, wear: dict) -> DiseaseResult:
    """Coronary Artery Disease risk."""
    score   = 0.0
    drivers = []

    ldl    = _get(lab, "ldl")
    hdl    = _get(lab, "hdl")
    trigs  = _get(lab, "triglycerides")
    sbp    = _get(wear, "systolic_bp") or _get(lab, "systolic_bp")
    hr     = _get(wear, "heart_rate")
    hrv    = _get(wear, "hrv")
    steps  = _get(wear, "steps")
    stress = _get(wear, "stress_index")
    hba1c  = _get(lab, "hba1c")

    if ldl is not None:
        if ldl > 190:   score += 30; drivers.append(f"LDL {ldl} mg/dL — very high atherosclerosis risk")
        elif ldl > 160: score += 18; drivers.append(f"LDL {ldl} mg/dL — elevated")
        elif ldl > 130: score += 8;  drivers.append(f"LDL {ldl} mg/dL — borderline high")

    if hdl is not None and hdl < 35:
        score += 15; drivers.append(f"Low HDL {hdl} mg/dL — cardioprotection reduced")
    elif hdl is not None and hdl < 40:
        score += 8;  drivers.append(f"Low-normal HDL {hdl} mg/dL")

    if trigs is not None and trigs > 200:
        score += 10; drivers.append(f"Triglycerides {trigs} mg/dL — elevated CV risk")

    if sbp is not None and sbp > 140:
        score += 10; drivers.append(f"Elevated BP {sbp} mmHg — arterial wall stress")

    if hrv is not None and hrv < 20:
        score += 10; drivers.append(f"Low HRV {hrv}ms — autonomic dysfunction signal")

    if hr is not None and hr > 100:
        score += 5;  drivers.append(f"Elevated resting HR {hr} bpm")

    if steps is not None and steps < 3000:
        score += 8;  drivers.append(f"Very low activity ({int(steps)} steps) — sedentary lifestyle")

    if stress is not None and stress > 75:
        score += 7;  drivers.append(f"High chronic stress ({stress}) — cortisol-linked plaque risk")

    if hba1c is not None and hba1c > 6.5:
        score += 7;  drivers.append(f"Diabetic HbA1c {hba1c}% — 2× CAD risk multiplier")

    advice = (
        "Adopt a heart-healthy diet (Mediterranean style). Aim for 150+ min/week of moderate "
        "exercise. Quit smoking if applicable. Request a lipid panel + ECG from your cardiologist."
    )
    return DiseaseResult("Coronary Artery Disease", _cap(score), _tier(_cap(score)), drivers, advice, "❤️")


def _anaemia(lab: dict, wear: dict) -> DiseaseResult:
    score   = 0.0
    drivers = []

    hgb  = _get(lab, "hemoglobin")
    rbc  = _get(lab, "rbc")
    mcv  = _get(lab, "mcv")
    mch  = _get(lab, "mch")
    rdw  = _get(lab, "rdw")
    hct  = _get(lab, "hematocrit")
    spo2 = _get(wear, "spo2")

    if hgb is not None:
        if hgb < 7:    score += 55; drivers.append(f"Severe anaemia: Hgb {hgb} g/dL")
        elif hgb < 10: score += 35; drivers.append(f"Moderate anaemia: Hgb {hgb} g/dL")
        elif hgb < 12: score += 20; drivers.append(f"Mild anaemia: Hgb {hgb} g/dL")
        elif hgb < 13.5: score += 8; drivers.append(f"Low-normal Hgb {hgb} g/dL (borderline)")

    if rbc is not None and rbc < 3.5:
        score += 12; drivers.append(f"Low RBC count {rbc} M/µL")

    if mcv is not None:
        if mcv < 75:   score += 10; drivers.append(f"Microcytic MCV {mcv} fL — possible iron deficiency")
        elif mcv > 100: score += 10; drivers.append(f"Macrocytic MCV {mcv} fL — B12/folate deficiency?")

    if rdw is not None and rdw > 15:
        score += 8;  drivers.append(f"High RDW {rdw}% — mixed or nutritional anaemia")

    if hct is not None and hct < 36:
        score += 8;  drivers.append(f"Low haematocrit {hct}%")

    if spo2 is not None and spo2 < 95:
        score += 10; drivers.append(f"Low SpO2 {spo2}% — oxygen delivery compromise")

    advice = (
        "Request iron studies, B12, and folate levels. Increase iron-rich foods (leafy greens, "
        "red meat, legumes). Severe anaemia (Hgb<8) requires urgent physician evaluation."
    )
    return DiseaseResult("Anaemia", _cap(score), _tier(_cap(score)), drivers, advice, "🔴")


def _ckd(lab: dict, wear: dict) -> DiseaseResult:
    """Chronic Kidney Disease."""
    score   = 0.0
    drivers = []

    creat = _get(lab, "creatinine")
    urea  = _get(lab, "urea")
    sbp   = _get(wear, "systolic_bp") or _get(lab, "systolic_bp")
    hba1c = _get(lab, "hba1c")

    if creat is not None:
        if creat > 5.0:   score += 55; drivers.append(f"Creatinine {creat} mg/dL — severe renal impairment")
        elif creat > 2.0: score += 35; drivers.append(f"Creatinine {creat} mg/dL — moderate CKD signal")
        elif creat > 1.4: score += 18; drivers.append(f"Creatinine {creat} mg/dL — mildly elevated")

    if urea is not None:
        if urea > 60:   score += 20; drivers.append(f"BUN/Urea {urea} mg/dL — severely elevated")
        elif urea > 30: score += 10; drivers.append(f"BUN/Urea {urea} mg/dL — elevated")

    if sbp is not None and sbp > 140:
        score += 8;  drivers.append(f"Hypertension {sbp} mmHg — major CKD progression driver")

    if hba1c is not None and hba1c > 7.0:
        score += 10; drivers.append(f"Poorly controlled diabetes (HbA1c {hba1c}%) — diabetic nephropathy risk")

    advice = (
        "Stay well hydrated. Avoid NSAIDs and nephrotoxic agents. Low-protein diet if creatinine "
        "> 2. Request eGFR and urine protein/creatinine ratio from your nephrologist."
    )
    return DiseaseResult("Chronic Kidney Disease", _cap(score), _tier(_cap(score)), drivers, advice, "🫘")


def _liver_disease(lab: dict, wear: dict) -> DiseaseResult:
    score   = 0.0
    drivers = []

    sgpt  = _get(lab, "sgpt")   # ALT
    sgot  = _get(lab, "sgot")   # AST
    trigs = _get(lab, "triglycerides")
    ldl   = _get(lab, "ldl")
    steps = _get(wear, "steps")
    sleep = _get(wear, "sleep_hours")

    if sgpt is not None:
        if sgpt > 200:  score += 40; drivers.append(f"ALT/SGPT {sgpt} U/L — severely elevated")
        elif sgpt > 80: score += 22; drivers.append(f"ALT/SGPT {sgpt} U/L — elevated (liver stress)")
        elif sgpt > 56: score += 10; drivers.append(f"ALT/SGPT {sgpt} U/L — above normal")

    if sgot is not None:
        if sgot > 120:  score += 20; drivers.append(f"AST/SGOT {sgot} U/L — severely elevated")
        elif sgot > 40: score += 10; drivers.append(f"AST/SGOT {sgot} U/L — elevated")

    # AST:ALT ratio >2 suggests alcoholic liver
    if sgot and sgpt and sgpt > 0 and (sgot / sgpt) > 2:
        score += 10; drivers.append(f"AST:ALT ratio {sgot/sgpt:.1f} > 2 — alcoholic hepatitis pattern")

    if trigs is not None and trigs > 200:
        score += 10; drivers.append(f"Triglycerides {trigs} mg/dL — NAFLD risk factor")

    if steps is not None and steps < 3000:
        score += 7;  drivers.append(f"Very low activity ({int(steps)} steps) — NAFLD progression risk")

    if sleep is not None and sleep < 5:
        score += 5;  drivers.append(f"Poor sleep ({sleep}h) — linked to fatty liver progression")

    advice = (
        "Avoid alcohol, processed foods, and unnecessary medications. Maintain healthy weight. "
        "Request liver function panel + ultrasound abdomen if SGPT > 2× upper limit of normal."
    )
    return DiseaseResult("Liver Disease (NAFLD)", _cap(score), _tier(_cap(score)), drivers, advice, "🟤")


def _thyroid(lab: dict, wear: dict) -> DiseaseResult:
    score    = 0.0
    drivers  = []
    disorder = "Thyroid Disorder"

    tsh  = _get(lab, "tsh")
    hr   = _get(wear, "heart_rate")
    hrv  = _get(wear, "hrv")
    sleep = _get(wear, "sleep_hours")
    steps = _get(wear, "steps")

    if tsh is not None:
        if tsh > 10:
            score += 45; drivers.append(f"TSH {tsh} mIU/L — overt hypothyroidism"); disorder = "Hypothyroidism"
        elif tsh > 4.5:
            score += 25; drivers.append(f"TSH {tsh} mIU/L — subclinical hypothyroidism"); disorder = "Subclinical Hypothyroidism"
        elif tsh < 0.1:
            score += 45; drivers.append(f"TSH {tsh} mIU/L — overt hyperthyroidism"); disorder = "Hyperthyroidism"
        elif tsh < 0.4:
            score += 25; drivers.append(f"TSH {tsh} mIU/L — subclinical hyperthyroidism"); disorder = "Subclinical Hyperthyroidism"

    if hr is not None and hr > 100 and tsh and tsh < 0.4:
        score += 10; drivers.append(f"Tachycardia {hr} bpm consistent with hyperthyroid state")
    if hr is not None and hr < 55 and tsh and tsh > 4.5:
        score += 10; drivers.append(f"Bradycardia {hr} bpm consistent with hypothyroid state")

    if steps is not None and steps < 3000 and tsh and tsh > 4.5:
        score += 8;  drivers.append(f"Low activity ({int(steps)} steps) — hypothyroid fatigue pattern")
    if sleep is not None and sleep > 9.5 and tsh and tsh > 4.5:
        score += 5;  drivers.append(f"Excessive sleep ({sleep}h) — hypothyroid fatigue pattern")

    advice = (
        "Request free T3, free T4, and anti-TPO antibodies. Thyroid disorders are very treatable. "
        "If TSH is abnormal, consult an endocrinologist for personalised management."
    )
    return DiseaseResult(disorder, _cap(score), _tier(_cap(score)), drivers, advice, "🦋")


def _sleep_apnea(lab: dict, wear: dict) -> DiseaseResult:
    score   = 0.0
    drivers = []

    spo2        = _get(wear, "spo2")
    hrv         = _get(wear, "hrv")
    sleep_hrs   = _get(wear, "sleep_hours")
    sleep_qual  = _get(wear, "sleep_quality")
    stress      = _get(wear, "stress_index")
    hr          = _get(wear, "heart_rate")
    sbp         = _get(wear, "systolic_bp") or _get(lab, "systolic_bp")

    if spo2 is not None:
        if spo2 < 90:   score += 40; drivers.append(f"SpO2 {spo2}% — severe nocturnal hypoxia")
        elif spo2 < 94: score += 22; drivers.append(f"SpO2 {spo2}% — borderline oxygen saturation")

    if hrv is not None and hrv < 15:
        score += 15; drivers.append(f"Very low HRV {hrv}ms — autonomic stress from fragmented sleep")

    if sleep_qual is not None and sleep_qual < 50:
        score += 12; drivers.append(f"Low sleep quality {sleep_qual}% — fragmentation pattern")

    if sleep_hrs is not None:
        if sleep_hrs > 9 and sleep_qual and sleep_qual < 60:
            score += 8;  drivers.append(f"Long but poor sleep ({sleep_hrs}h, {sleep_qual}%) — unrefreshing sleep")
        elif sleep_hrs < 5:
            score += 6;  drivers.append(f"Short sleep ({sleep_hrs}h) — possibly apnea-interrupted")

    if stress is not None and stress > 70:
        score += 7;  drivers.append(f"High stress ({stress}) — sympathetic arousal disrupting sleep")

    if sbp is not None and sbp > 140:
        score += 8;  drivers.append(f"Hypertension {sbp} mmHg — associated with untreated OSA")

    advice = (
        "Consider a home sleep study (polysomnography). Avoid alcohol before sleep, "
        "sleep on your side, and maintain a healthy weight. If suspected, consult a pulmonologist."
    )
    return DiseaseResult("Sleep Apnea / Sleep Disorder", _cap(score), _tier(_cap(score)), drivers, advice, "😴")


def _metabolic_syndrome(lab: dict, wear: dict) -> DiseaseResult:
    score    = 0.0
    drivers  = []
    criteria = 0   # MetS is diagnosed when ≥3 of 5 NCEP criteria are met

    sbp    = _get(wear, "systolic_bp") or _get(lab, "systolic_bp")
    trigs  = _get(lab, "triglycerides")
    hdl    = _get(lab, "hdl")
    fbs    = _get(lab, "fasting_glucose")
    steps  = _get(wear, "steps")
    stress = _get(wear, "stress_index")

    if sbp is not None and sbp >= 130:
        criteria += 1; score += 20; drivers.append(f"Elevated BP {sbp} mmHg (MetS criterion)")

    if trigs is not None and trigs >= 150:
        criteria += 1; score += 20; drivers.append(f"High triglycerides {trigs} mg/dL (MetS criterion)")

    if hdl is not None and hdl < 40:
        criteria += 1; score += 20; drivers.append(f"Low HDL {hdl} mg/dL (MetS criterion)")

    if fbs is not None and fbs >= 100:
        criteria += 1; score += 20; drivers.append(f"Elevated fasting glucose {fbs} mg/dL (MetS criterion)")

    # Waist circumference not measurable — use step count as sedentary proxy
    if steps is not None and steps < 4000:
        criteria += 1; score += 10; drivers.append(f"Very low activity ({int(steps)} steps) — obesity proxy")

    if criteria >= 3:
        score = max(score, 65)
        drivers.insert(0, f"⚠️  {criteria}/5 Metabolic Syndrome criteria met")

    if stress is not None and stress > 65:
        score += 5; drivers.append(f"Chronic stress ({stress}) — cortisol-driven metabolic dysfunction")

    advice = (
        "Metabolic syndrome significantly raises risk of heart disease and diabetes. "
        "Focus on weight loss, 150+ min/week of aerobic exercise, and reduced refined carb intake. "
        "Request full lipid panel + fasting insulin from your physician."
    )
    return DiseaseResult("Metabolic Syndrome", _cap(score), _tier(_cap(score)), drivers, advice, "⚙️")


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_SCORERS = [
    _diabetes,
    _hypertension,
    _cad,
    _anaemia,
    _ckd,
    _liver_disease,
    _thyroid,
    _sleep_apnea,
    _metabolic_syndrome,
]


def predict_diseases(lab: dict, wear: dict,
                     threshold: float = 20.0) -> list[DiseaseResult]:
    """
    Run all disease scorers against lab + wearable data.

    Returns list of DiseaseResult sorted by risk_pct descending,
    filtered to those with risk_pct >= threshold.
    """
    results = []
    for scorer in DISEASE_SCORERS:
        try:
            result = scorer(lab, wear)
            if result.risk_pct >= threshold:
                results.append(result)
        except Exception:
            continue
    return sorted(results, key=lambda r: r.risk_pct, reverse=True)


def format_disease_report(results: list[DiseaseResult]) -> str:
    """Returns a console-printable report string."""
    if not results:
        return "  ✅ No significant disease risk flags detected at current threshold."

    lines = []
    for r in results:
        tier_icon = {"Low": "🟢", "Moderate": "🟡", "High": "🔴", "Critical": "🚨"}.get(r.risk_tier, "⚪")
        lines.append(f"\n  {r.icon} {r.disease}")
        lines.append(f"     Risk: {tier_icon} {r.risk_tier}  ({r.risk_pct:.0f}/100)")
        if r.key_drivers:
            lines.append("     Key drivers:")
            for d in r.key_drivers[:4]:
                lines.append(f"       • {d}")
        lines.append(f"     💡 {r.advice}")
    return "\n".join(lines)


def diseases_to_json(results: list[DiseaseResult]) -> list[dict]:
    """Serialise results for JSON dashboard export."""
    return [
        {
            "disease":    r.disease,
            "risk_pct":   r.risk_pct,
            "risk_tier":  r.risk_tier,
            "icon":       r.icon,
            "drivers":    r.key_drivers,
            "advice":     r.advice,
        }
        for r in results
    ]
