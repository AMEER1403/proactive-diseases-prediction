"""
api_server.py
─────────────────────────────────────────────────────────────────────────────
Flask REST API — bridges the Health Intelligence System to mobile/web apps.

Run:
    pip install flask flask-cors
    python api_server.py          (development mode, port 5000)

Endpoints:
  POST /api/register              → create new user
  POST /api/login                 → login by patient ID
  GET  /api/user/<user_id>        → get profile + visit history
  POST /api/analyze/<user_id>     → run full analysis (JSON body: lab + wearable)
  GET  /api/report/<user_id>      → fetch latest report_data.json for that user
  GET  /api/fitbit/status         → check if Fitbit credentials configured
  POST /api/fitbit/sync           → trigger Fitbit data fetch (returns data)
  GET  /api/diseases/<user_id>    → disease predictions for latest visit

All responses: { "ok": bool, "data": ..., "error": str|null }
─────────────────────────────────────────────────────────────────────────────
"""

import os, json, uuid, pickle
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Import your system modules ────────────────────────────────────────────────
from disease_engine import predict_diseases, diseases_to_json
import fitbit_api

# Re-use constants from med.py without running main()
USERS_ROOT   = "users"
GLOBAL_MODEL = "model_cache.pkl"

app = Flask(__name__)
CORS(app)   # allow cross-origin requests from any frontend


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS  (duplicated here to keep api_server.py self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _ok(data=None):
    return jsonify({"ok": True, "data": data, "error": None})

def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "data": None, "error": msg}), code

def _users_index_path():
    return os.path.join(USERS_ROOT, "users_index.json")

def _load_index():
    p = _users_index_path()
    return json.load(open(p)) if os.path.exists(p) else {}

def _save_index(idx):
    os.makedirs(USERS_ROOT, exist_ok=True)
    json.dump(idx, open(_users_index_path(), "w"), indent=2)

def _user_dir(uid): return os.path.join(USERS_ROOT, uid)
def _user_file(uid, fn): return os.path.join(_user_dir(uid), fn)

def _load_history(uid):
    p = _user_file(uid, "health_history.json")
    return json.load(open(p)) if os.path.exists(p) else []

def _load_report(uid):
    p = _user_file(uid, "report_data.json")
    return json.load(open(p)) if os.path.exists(p) else None

def _predictor():
    """Load the shared ML predictor (lazy, cached in app context)."""
    if not hasattr(app, "_predictor"):
        if os.path.exists(GLOBAL_MODEL):
            with open(GLOBAL_MODEL, "rb") as f:
                app._predictor = pickle.load(f)
        else:
            return None
    return app._predictor


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register():
    """
    Body: { "name": "Full Name" }
    Returns: { "user_id": "PSNA-XXXX-XXXX", "name": "Full Name", "created_at": "..." }
    """
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return _err("name is required")

    uid     = uuid.uuid4().hex[:8].upper()
    user_id = f"PSNA-{uid[:4]}-{uid[4:]}"
    profile = {"name": name, "created_at": datetime.now().isoformat()}

    os.makedirs(_user_dir(user_id), exist_ok=True)
    json.dump(profile, open(_user_file(user_id, "profile.json"), "w"), indent=2)

    idx = _load_index()
    idx[user_id] = profile
    _save_index(idx)

    return _ok({"user_id": user_id, **profile})


@app.route("/api/login", methods=["POST"])
def login():
    """
    Body: { "user_id": "PSNA-XXXX-XXXX" }
    Returns: profile + visit count
    """
    body    = request.get_json(silent=True) or {}
    uid     = (body.get("user_id") or "").strip().upper()
    index   = _load_index()
    if uid not in index:
        return _err("Patient ID not found", 404)
    history  = _load_history(uid)
    return _ok({
        "user_id":    uid,
        "name":       index[uid]["name"],
        "created_at": index[uid]["created_at"],
        "visit_count": len(history),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  USER DATA ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/user/<user_id>", methods=["GET"])
def get_user(user_id):
    """Full profile + complete visit history."""
    uid   = user_id.upper()
    index = _load_index()
    if uid not in index:
        return _err("User not found", 404)
    return _ok({
        "profile": index[uid],
        "history": _load_history(uid),
        "latest_report": _load_report(uid),
    })


@app.route("/api/report/<user_id>", methods=["GET"])
def get_report(user_id):
    """Latest report_data.json for this patient."""
    uid    = user_id.upper()
    report = _load_report(uid)
    if report is None:
        return _err("No report found for this user", 404)
    return _ok(report)


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze/<user_id>", methods=["POST"])
def analyze(user_id):
    """
    Run a full health analysis for a patient from mobile/web app.

    Body: {
      "lab":      { "hemoglobin": 14.2, "wbc": 6.5, ... },
      "wearable": { "heart_rate": 78,   "spo2": 98, ... }
    }

    Returns full prediction + disease forecast + updated history.
    """
    uid   = user_id.upper()
    index = _load_index()
    if uid not in index:
        return _err("User not found", 404)

    body     = request.get_json(silent=True) or {}
    lab_data = body.get("lab", {})
    wear_data = body.get("wearable", {})

    if not lab_data and not wear_data:
        return _err("At least one of 'lab' or 'wearable' data is required")

    predictor = _predictor()
    if predictor is None:
        return _err("ML model not found — run med.py first to train and cache the model", 503)

    import numpy as np
    from med import ALL_FEATURES, NORMAL_RANGES, RISK_LABELS

    # Build feature vector
    combined = {**lab_data, **wear_data}
    row = []
    for feat in ALL_FEATURES:
        v = combined.get(feat)
        if v is not None:
            row.append(float(v))
        else:
            lo, hi = NORMAL_RANGES.get(feat, (0, 1))
            row.append((lo + hi) / 2)
    fv = np.array(row, dtype=float).reshape(1, -1)

    idx_pred, score = predictor.predict(fv)
    risk_label      = RISK_LABELS[idx_pred]
    explanations    = predictor.explain(fv)

    # Disease prediction
    diseases = predict_diseases(lab_data, wear_data)
    disease_json = diseases_to_json(diseases)

    # Log to history
    history = _load_history(uid)
    entry = {
        "timestamp":  datetime.now().isoformat(),
        "risk_score": score,
        "risk_label": risk_label,
        "lab":        lab_data,
        "wearable":   wear_data,
        "diseases":   disease_json,
    }
    history.append(entry)
    json.dump(history, open(_user_file(uid, "health_history.json"), "w"), indent=2)

    return _ok({
        "user_id":     uid,
        "name":        index[uid]["name"],
        "risk_label":  risk_label,
        "risk_score":  score,
        "explanations": explanations,
        "diseases":    disease_json,
        "timestamp":   entry["timestamp"],
    })


# ─────────────────────────────────────────────────────────────────────────────
#  DISEASE PREDICTION ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/diseases/<user_id>", methods=["GET"])
def get_diseases(user_id):
    """Disease predictions derived from the user's latest visit data."""
    uid     = user_id.upper()
    history = _load_history(uid)
    if not history:
        return _err("No analysis data found for this user", 404)

    latest   = history[-1]
    diseases = predict_diseases(latest.get("lab", {}), latest.get("wearable", {}))
    return _ok(diseases_to_json(diseases))


# ─────────────────────────────────────────────────────────────────────────────
#  FITBIT ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/fitbit/status", methods=["GET"])
def fitbit_status():
    configured = fitbit_api.FITBIT_CLIENT_ID != "YOUR_CLIENT_ID"
    token      = fitbit_api._load_token()
    return _ok({
        "configured":  configured,
        "token_saved": token is not None,
        "token_valid": (configured and token is not None and fitbit_api._is_token_valid(token)),
    })


@app.route("/api/fitbit/sync", methods=["POST"])
def fitbit_sync():
    """
    Triggers a Fitbit data fetch.  If running headless (no browser), the
    OAuth flow can't complete — call this from a desktop environment only.
    Returns whatever metrics were retrieved.
    """
    data = fitbit_api.get_fitbit_data()
    if data is None:
        return _err("Fitbit sync failed — check credentials or complete OAuth2 flow on desktop")
    return _ok(data)


# ─────────────────────────────────────────────────────────────────────────────
#  ALL USERS (admin)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/users", methods=["GET"])
def list_users():
    """List all registered patients (for admin/clinic dashboard)."""
    index = _load_index()
    result = []
    for uid, info in index.items():
        h = _load_history(uid)
        result.append({
            "user_id":     uid,
            "name":        info["name"],
            "created_at":  info["created_at"],
            "visit_count": len(h),
            "last_visit":  h[-1]["timestamp"][:10] if h else None,
            "last_risk":   h[-1]["risk_label"]     if h else None,
        })
    return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Health Intelligence API running at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
