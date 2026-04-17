"""
=============================================================================
  MULTIMODAL EXPLAINABLE AI-BASED PREDICTIVE HEALTH INTELLIGENCE SYSTEM
  Integrated Biochemical and Clinical Determinant Analysis for
  Proactive Disease Risk Prediction
=============================================================================
  Authors : Harish D V | Mohammed Ameer C | Prabanja Kumar R
  Dept    : AI & Data Science, PSNA College of Engineering and Technology
=============================================================================

SETUP (run once):
    pip install -r requirements.txt
    pip install flask flask-cors        ← for API server (app development)

REQUIRES:
    Tesseract OCR  → https://github.com/UB-Mannheim/tesseract/wiki
    Poppler        → https://github.com/oschwartz10612/poppler-windows/releases
    Ollama + llama3.2:1b → https://ollama.com  →  ollama pull llama3.2:1b

FITBIT SETUP (optional):
    1. Register at https://dev.fitbit.com/apps/new
    2. Set Redirect URI: http://localhost:8765/callback
    3. Set FITBIT_CLIENT_ID and FITBIT_CLIENT_SECRET in fitbit_api.py
       or via environment variables.

APP DEVELOPMENT:
    Run the REST API alongside this script:
        python api_server.py
    Then connect your mobile/web app to http://localhost:5000/api/...
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re, os, json, random, warnings, pickle, webbrowser, uuid
import numpy as np
from datetime import datetime
from collections import deque

from pdf2image import convert_from_path
import pytesseract

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from colorama import Fore, Style, init
init(autoreset=True)
warnings.filterwarnings("ignore")

# ── Disease engine & Fitbit (new modules) ────────────────────────────────────
from disease_engine import predict_diseases, format_disease_report, diseases_to_json
import fitbit_api

# ─────────────────────────────────────────────────────────────────────────────
#  ★  CHANGE THESE PATHS  ★
# ─────────────────────────────────────────────────────────────────────────────
TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH    = r"C:\poppler\poppler-25.12.0\Library\bin"
# PDF_PATH is no longer hardcoded — users upload their report interactively

USERS_ROOT      = "users"
DASHBOARD_HTML  = "results_dashboard.html"
GLOBAL_MODEL    = "model_cache.pkl"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
LAB_FEATURES = [
    "hemoglobin", "wbc", "platelets",
    "ldl", "hdl", "triglycerides",
    "fasting_glucose", "hba1c",
    "creatinine", "urea",
    "sgot", "sgpt", "tsh",
    "rbc", "hematocrit", "mch", "mcv", "mchc", "rdw",
    "neutrophils", "lymphocytes", "monocytes", "eosinophils",
]
WEARABLE_FEATURES = [
    "heart_rate", "hrv", "spo2",
    "systolic_bp", "diastolic_bp",
    "sleep_hours", "sleep_quality",
    "steps", "stress_index",
]
ALL_FEATURES = LAB_FEATURES + WEARABLE_FEATURES
RISK_LABELS  = ["Low", "Moderate", "High", "Critical"]
RISK_COLORS  = [Fore.GREEN, Fore.YELLOW, Fore.RED, Fore.MAGENTA]

NORMAL_RANGES = {
    "hemoglobin":     (13.5, 17.5),
    "wbc":            (3.5,  10.5),
    "platelets":      (150000, 450000),
    "rbc":            (4.3,  5.7),
    "hematocrit":     (39.0, 50.0),
    "mch":            (26.0, 34.0),
    "mcv":            (81.0, 95.0),
    "mchc":           (31.0, 36.0),
    "rdw":            (11.8, 15.6),
    "neutrophils":    (1700, 7000),
    "lymphocytes":    (900,  2900),
    "monocytes":      (300,  900),
    "eosinophils":    (50,   500),
    "ldl":            (0,    100),
    "hdl":            (40,   90),
    "triglycerides":  (0,    150),
    "fasting_glucose":(70,   100),
    "hba1c":          (4.0,  5.6),
    "creatinine":     (0.6,  1.2),
    "urea":           (7,    20),
    "sgot":           (10,   40),
    "sgpt":           (7,    56),
    "tsh":            (0.4,  4.0),
    "heart_rate":     (60,   100),
    "hrv":            (20,   70),
    "spo2":           (95,   100),
    "systolic_bp":    (90,   120),
    "diastolic_bp":   (60,   80),
    "sleep_hours":    (7,    9),
    "sleep_quality":  (70,   100),
    "steps":          (7000, 12000),
    "stress_index":   (0,    40),
}

# ─────────────────────────────────────────────────────────────────────────────
#  USER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _users_index_path() -> str:
    return os.path.join(USERS_ROOT, "users_index.json")

def _load_users_index() -> dict:
    path = _users_index_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def _save_users_index(index: dict):
    os.makedirs(USERS_ROOT, exist_ok=True)
    with open(_users_index_path(), "w") as f:
        json.dump(index, f, indent=2)

def user_dir(user_id: str) -> str:
    return os.path.join(USERS_ROOT, user_id)

def user_file(user_id: str, filename: str) -> str:
    return os.path.join(user_dir(user_id), filename)

def _create_new_user() -> tuple[str, str]:
    print(Fore.CYAN + "\n👤 NEW USER REGISTRATION")
    divider()
    while True:
        name = input(Fore.WHITE + "  Enter your full name: ").strip()
        if name:
            break
        print(Fore.RED + "  ❌ Name cannot be empty.")

    uid     = uuid.uuid4().hex[:8].upper()
    user_id = f"PSNA-{uid[:4]}-{uid[4:]}"

    os.makedirs(user_dir(user_id), exist_ok=True)
    profile = {"name": name, "created_at": datetime.now().isoformat()}
    with open(user_file(user_id, "profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    index = _load_users_index()
    index[user_id] = profile
    _save_users_index(index)

    print(Fore.GREEN + f"\n  ✅ Welcome, {name}!")
    print(Fore.GREEN + f"  🆔 Your unique Patient ID : {Style.BRIGHT}{user_id}")
    print(Fore.YELLOW + "  ⚠️  Please save this ID — you'll need it for future visits.\n")
    return user_id, name


def _login_existing_user() -> tuple[str, str]:
    index = _load_users_index()
    if not index:
        print(Fore.YELLOW + "\n  ⚠️  No users registered yet. Switching to new user registration.\n")
        return _create_new_user()

    print(Fore.CYAN + "\n🔑 RETURNING USER LOGIN")
    divider()
    print(Fore.CYAN + "  Registered patients on this system:")
    for uid, info in index.items():
        created = info.get("created_at", "")[:10]
        hist_path = user_file(uid, "health_history.json")
        n = len(json.load(open(hist_path))) if os.path.exists(hist_path) else 0
        print(Fore.WHITE + f"    {uid}  —  {info['name']}  (joined {created}, {n} visit(s))")
    print()

    while True:
        entered = input(Fore.WHITE + "  Enter your Patient ID: ").strip().upper()
        if entered in index:
            name      = index[entered]["name"]
            hist_path = user_file(entered, "health_history.json")
            n_visits  = len(json.load(open(hist_path))) if os.path.exists(hist_path) else 0
            print(Fore.GREEN + f"\n  ✅ Welcome back, {name}! ({n_visits} visit(s) on record)\n")
            return entered, name
        else:
            print(Fore.RED + f"  ❌ ID '{entered}' not found. Try again or press Ctrl+C.")


def select_user() -> tuple[str, str, bool]:
    """Returns (user_id, user_name, is_new_user)."""
    divider("PATIENT IDENTIFICATION")
    print(Fore.WHITE + "  [1] New patient  —  Register and get a unique Patient ID")
    print(Fore.WHITE + "  [2] Returning patient  —  Login with existing Patient ID")
    print()
    while True:
        choice = input(Fore.WHITE + "  Select option (1 or 2): ").strip()
        if choice == "1":
            uid, name = _create_new_user()
            return uid, name, True
        elif choice == "2":
            uid, name = _login_existing_user()
            return uid, name, False
        else:
            print(Fore.RED + "  ❌ Please enter 1 or 2.")


# ─────────────────────────────────────────────────────────────────────────────
#  1. LAB REPORT READER  — PDF + any image format, with per-user storage
# ─────────────────────────────────────────────────────────────────────────────

# Supported upload formats
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
PDF_EXT    = ".pdf"

LAB_REPORTS_SUBDIR = "lab_reports"   # subfolder inside each user's directory


def _ocr_pdf(file_path: str) -> str:
    """Run Tesseract OCR on every page of a PDF."""
    text = ""
    try:
        print(Fore.CYAN + "   📄 Detected: PDF — converting pages to images...")
        images = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
        print(Fore.CYAN + f"   ✅ {len(images)} page(s) found")
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, config="--oem 3 --psm 6 -l eng")
            text += page_text + "\n"
            print(Fore.CYAN + f"   🔍 Page {i+1}: {len(page_text)} chars extracted")
    except Exception as e:
        print(Fore.RED + f"   ❌ PDF OCR Error: {e}")
    return text.strip()


def _ocr_image(file_path: str) -> str:
    """Run Tesseract OCR directly on an image file (any supported format)."""
    from PIL import Image
    text = ""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        print(Fore.CYAN + f"   🖼️  Detected: Image ({ext.upper().lstrip('.')}) — running OCR...")
        img  = Image.open(file_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 6 -l eng")
        print(Fore.CYAN + f"   ✅ {len(text)} characters extracted")
    except Exception as e:
        print(Fore.RED + f"   ❌ Image OCR Error: {e}")
    return text.strip()


def read_report_file(file_path: str) -> str:
    """Dispatch to correct OCR method. Returns extracted text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == PDF_EXT:
        return _ocr_pdf(file_path)
    elif ext in IMAGE_EXTS:
        return _ocr_image(file_path)
    else:
        print(Fore.RED + f"   ❌ Unsupported file format: '{ext}'")
        supported = f"PDF, {', '.join(sorted(s.lstrip('.').upper() for s in IMAGE_EXTS))}"
        print(Fore.YELLOW + f"      Supported: {supported}")
        return ""


def _save_report_to_user_folder(user_id: str, src_path: str) -> str:
    """Copy uploaded file into users/<user_id>/lab_reports/ with timestamp prefix."""
    import shutil
    reports_dir = os.path.join(user_dir(user_id), LAB_REPORTS_SUBDIR)
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext       = os.path.splitext(src_path)[1].lower()
    dest_name = f"report_{timestamp}{ext}"
    dest_path = os.path.join(reports_dir, dest_name)
    shutil.copy2(src_path, dest_path)
    return dest_path


def _list_saved_reports(user_id: str) -> list:
    """Return sorted list of previously saved report paths for this user."""
    reports_dir = os.path.join(user_dir(user_id), LAB_REPORTS_SUBDIR)
    if not os.path.exists(reports_dir):
        return []
    all_exts = IMAGE_EXTS | {PDF_EXT}
    return sorted([
        os.path.join(reports_dir, f)
        for f in os.listdir(reports_dir)
        if os.path.splitext(f)[1].lower() in all_exts
    ])


def _prompt_file_path() -> str:
    """Prompt user to enter/paste the path to their lab report file."""
    supported = f"PDF, {', '.join(sorted(s.lstrip('.').upper() for s in IMAGE_EXTS))}"
    all_exts  = IMAGE_EXTS | {PDF_EXT}
    print(Fore.CYAN + f"\n  Supported formats: {supported}")
    print(Fore.YELLOW + "  Tip: drag-and-drop the file into this terminal to paste its path.\n")
    while True:
        raw = input(Fore.WHITE + "  Enter full path to lab report: ").strip().strip('\'\" ')
        if not raw:
            print(Fore.RED + "  ❌ Path cannot be empty.")
            continue
        if not os.path.isfile(raw):
            print(Fore.RED + f"  ❌ File not found: {raw}")
            continue
        ext = os.path.splitext(raw)[1].lower()
        if ext not in all_exts:
            print(Fore.RED + f"  ❌ Unsupported format '{ext}'. Supported: {supported}")
            continue
        return raw


def get_lab_report(user_id: str, is_new_user: bool):
    """
    Handle lab report upload or reuse for a user session.

    - New user / no saved reports  → must upload
    - Returning user with reports  → upload new OR pick a saved one

    Returns (lab_data dict, source description string).
    """
    divider("LAB REPORT")

    saved_reports = _list_saved_reports(user_id)
    has_saved     = len(saved_reports) > 0
    use_existing  = False
    chosen_path   = None

    if is_new_user or not has_saved:
        if not has_saved and not is_new_user:
            print(Fore.YELLOW + "  ℹ️  No previous lab report found for your account.")
        print(Fore.WHITE + "  Please upload your lab report to continue.\n")

    else:
        # Returning user with at least one saved report
        latest      = saved_reports[-1]
        latest_name = os.path.basename(latest)
        try:
            ts_part  = latest_name.split("_", 1)[1].rsplit(".", 1)[0]
            ts_dt    = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
            ts_label = ts_dt.strftime("%d %b %Y at %H:%M")
        except Exception:
            ts_label = latest_name

        print(Fore.CYAN + f"  📋 Last report on file: {ts_label}  ({latest_name})")
        print()
        print(Fore.WHITE + "  [1] 📤 Upload a new lab report")
        print(Fore.WHITE + f"  [2] 🔄 Use the last saved report  ({ts_label})")
        if len(saved_reports) > 1:
            print(Fore.WHITE + f"  [3] 📂 Choose from all saved reports  ({len(saved_reports)} on file)")
        print()

        max_opt = 3 if len(saved_reports) > 1 else 2
        while True:
            ch = input(Fore.WHITE + f"  Select option (1–{max_opt}): ").strip()
            if ch == "1":
                break
            elif ch == "2":
                use_existing = True
                chosen_path  = latest
                break
            elif ch == "3" and len(saved_reports) > 1:
                print(Fore.CYAN + "\n  All saved reports:")
                for n, p in enumerate(saved_reports, 1):
                    fname = os.path.basename(p)
                    try:
                        ts_part = fname.split("_", 1)[1].rsplit(".", 1)[0]
                        ts_dt   = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
                        label   = ts_dt.strftime("%d %b %Y at %H:%M")
                    except Exception:
                        label = fname
                    print(Fore.WHITE + f"    [{n}] {label}  ({fname})")
                print()
                while True:
                    sel = input(Fore.WHITE + f"  Pick report number (1–{len(saved_reports)}): ").strip()
                    if sel.isdigit() and 1 <= int(sel) <= len(saved_reports):
                        chosen_path  = saved_reports[int(sel) - 1]
                        use_existing = True
                        break
                    print(Fore.RED + f"  ❌ Enter a number between 1 and {len(saved_reports)}")
                break
            else:
                print(Fore.RED + f"  ❌ Enter 1–{max_opt}.")

    # ── OCR ──────────────────────────────────────────────────────────────────
    if use_existing:
        print(Fore.CYAN + f"\n  📂 Reading saved report: {os.path.basename(chosen_path)}")
        raw_text = read_report_file(chosen_path)
        source   = f"Saved: {os.path.basename(chosen_path)}"
    else:
        file_path = _prompt_file_path()
        print(Fore.CYAN + f"\n📂 Reading lab report: {os.path.basename(file_path)}")
        raw_text  = read_report_file(file_path)
        if raw_text:
            saved_to = _save_report_to_user_folder(user_id, file_path)
            print(Fore.GREEN + f"   💾 Report saved to your account → {os.path.basename(saved_to)}")
        source = f"Uploaded: {os.path.basename(file_path)}"

    if not raw_text:
        print(Fore.RED + "❌ Could not extract text from the report.")
        print(Fore.YELLOW + "   Check Tesseract and Poppler installation.")
        return {}, source

    lab_data = extract_lab_parameters(raw_text)

    if not lab_data:
        print(Fore.YELLOW + "  ⚠️  No standard lab values detected.")
        print(Fore.YELLOW + "     Analysis will proceed with wearable data only.")
    else:
        print(Fore.GREEN + f"   ✅ {len(lab_data)} lab parameter(s) extracted")

    return lab_data, source

# ─────────────────────────────────────────────────────────────────────────────
#  2. PARAMETER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def normalize_number(raw: str, sanity_lo=None, sanity_hi=None) -> float:
    s = re.sub(r"\s", "", raw)
    has_comma, has_dot = "," in s, "." in s
    if has_comma and has_dot:
        dp, cp = s.index("."), s.index(",")
        s = s.replace(".", "").replace(",", ".") if dp < cp else s.replace(",", "")
        return float(s)
    elif has_comma and not has_dot:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 3:
            as_int, as_dec = float(s.replace(",", "")), float(s.replace(",", "."))
            if sanity_lo is not None and sanity_lo <= as_dec <= sanity_hi:
                return as_dec
            return as_int
        return float(s.replace(",", "."))
    elif has_dot and not has_comma:
        parts = s.split(".")
        if len(parts) == 2 and len(parts[1]) == 3:
            as_int, as_dec = float(s.replace(".", "")), float(s)
            if sanity_lo is not None:
                dec_ok = sanity_lo <= as_dec <= sanity_hi
                int_ok = sanity_lo <= as_int <= sanity_hi
                if dec_ok and not int_ok: return as_dec
                if int_ok and not dec_ok: return as_int
            return as_dec
        return float(s)
    return float(s)

NUM = r"([\d][\d\s,\.]*[\d]|[\d])"

LAB_PATTERNS = {
    "hemoglobin":     [r"hemo(?:glo(?:bin|g(?:lo)?in|s\s*in)?)\s*[:\-]?\s*" + NUM,
                       r"hemoglob\S*\s*[:\-]?\s*" + NUM],
    "rbc":            [r"red\s+blood\s+cell[s]?\s*[:\-]?\s*" + NUM],
    "hematocrit":     [r"hematocrit\s*[:\-]?\s*" + NUM],
    "mch":            [r"mean\s+corpuscular\s+hemo[^\n]*?mchn?\)?\s*[:\-]?\s*" + NUM,
                       r"\bmch\b\s*[:\-]?\s*" + NUM],
    "mcv":            [r"mean\s+corpuscular\s+vol[^\n]*?mcvn?\)?\s*[:\-]?\s*" + NUM,
                       r"\bmcv\b\s*[:\-]?\s*" + NUM],
    "mchc":           [r"concentration\s+\(mchc\)\s*[:©\-]?\s*" + NUM,
                       r"\bmchc\b\s*[:\-]?\s*" + NUM],
    "rdw":            [r"(?:red\s+cell\s+dist[^\n]*?rdw?\)?|rdw)\s*[:\-=]?\s*" + NUM],
    "wbc":            [r"white\s+blood\s+cell[s]?\s+" + NUM, r"\bwbc\b\s*[:\-]?\s*" + NUM],
    "neutrophils":    [r"neutrophil[s]?\s*[:\-]?\s*[\d.,%\s]*?\s+" + NUM],
    "lymphocytes":    [r"lymphocyte[s]?\s*[:\-]?\s*[\d.,%\s]*?\s+" + NUM],
    "monocytes":      [r"monocyte[s]?\s*[:\-]?\s*[\d.,%\s]*?\s+" + NUM],
    "eosinophils":    [r"eosinophil[s]?\s*[:\-]?\s*[\d.,%\s]*?\s+" + NUM],
    "platelets":      [r"total\s+platelet\s+count[:\-]?\s*" + NUM,
                       r"platelet[s]?\s*[:\-]?\s*" + NUM],
    "ldl":            [r"\bldl\b\s*[:\-]?\s*" + NUM],
    "hdl":            [r"\bhdl\b\s*[:\-]?\s*" + NUM],
    "triglycerides":  [r"triglyceride[s]?\s*[:\-]?\s*" + NUM],
    "fasting_glucose":[r"(?:fasting\s*(?:blood\s*)?glucose|fbg|fbs)\s*[:\-]?\s*" + NUM],
    "hba1c":          [r"hba1c\s*[:\-]?\s*" + NUM],
    "creatinine":     [r"creatinine\s*[:\-]?\s*" + NUM],
    "urea":           [r"(?:\burea\b|\bbun\b)\s*[:\-]?\s*" + NUM],
    "sgot":           [r"(?:sgot|ast)\s*[:\-]?\s*" + NUM],
    "sgpt":           [r"(?:sgpt|alt)\s*[:\-]?\s*" + NUM],
    "tsh":            [r"\btsh\b\s*[:\-]?\s*" + NUM],
    "heart_rate":     [r"(?:heart\s*rate|pulse)\s*[:\-]?\s*" + NUM + r"\s*(?:bpm)?"],
    "systolic_bp":    [r"(?:systolic|sbp)\s*[:\-]?\s*" + NUM],
    "diastolic_bp":   [r"(?:diastolic|dbp)\s*[:\-]?\s*" + NUM],
    "spo2":           [r"(?:spo2|oxygen\s*sat[^\s]*)\s*[:\-]?\s*" + NUM],
}

BP_PATTERN = re.compile(r"\b(\d{2,3})/(\d{2,3})\b")

SANITY = {
    "hemoglobin": (4,25), "rbc": (1,9), "hematocrit": (10,70), "wbc": (0.5,30),
    "platelets": (10000,800000), "neutrophils": (100,15000), "lymphocytes": (100,10000),
    "monocytes": (50,3000), "eosinophils": (0,3000), "mch": (10,50), "mcv": (50,130),
    "mchc": (20,45), "rdw": (5,30), "ldl": (20,400), "hdl": (10,120),
    "triglycerides": (20,1000), "fasting_glucose": (40,600), "hba1c": (3,15),
    "creatinine": (0.2,15), "urea": (2,200), "sgot": (5,2000), "sgpt": (5,2000),
    "tsh": (0.01,50), "heart_rate": (30,200), "systolic_bp": (60,250),
    "diastolic_bp": (30,150), "spo2": (60,100),
}

def extract_lab_parameters(raw_text: str) -> dict:
    text = raw_text.lower()
    extracted = {}
    for param, patterns in LAB_PATTERNS.items():
        san_lo, san_hi = SANITY.get(param, (None, None))
        for pat in patterns:
            try:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    value = normalize_number(m.group(1), san_lo, san_hi)
                    if san_lo is not None and not (san_lo <= value <= san_hi):
                        continue
                    extracted[param] = round(value, 2)
                    break
            except Exception:
                continue
    if "systolic_bp" not in extracted:
        m = BP_PATTERN.search(raw_text)
        if m:
            sbp, dbp = float(m.group(1)), float(m.group(2))
            if 60 <= sbp <= 250: extracted["systolic_bp"] = sbp
            if 30 <= dbp <= 150: extracted["diastolic_bp"] = dbp
    return extracted

# ─────────────────────────────────────────────────────────────────────────────
#  WEARABLE DATA — FITBIT FIRST, THEN MANUAL FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
WEARABLE_CSV = "wearable_data.csv"

WEARABLE_PROMPTS = {
    "heart_rate":    ("Heart rate",                  "bpm",   40,  200),
    "hrv":           ("Heart rate variability (HRV)","ms",    5,   120),
    "spo2":          ("Blood oxygen (SpO2)",          "%",    70,  100),
    "systolic_bp":   ("Systolic BP",                 "mmHg",  60,  250),
    "diastolic_bp":  ("Diastolic BP",                "mmHg",  30,  150),
    "sleep_hours":   ("Sleep last night",            "hrs",   0,   24),
    "sleep_quality": ("Sleep quality",               "%",     0,   100),
    "steps":         ("Steps today",                 "steps", 0,   50000),
    "stress_index":  ("Stress level",                "0–100", 0,   100),
}

def _prompt_missing_wearable(data: dict) -> dict:
    """Asks the user to fill in only the wearable fields missing from data."""
    missing = [p for p in WEARABLE_PROMPTS if p not in data or data[p] is None]
    if not missing:
        return data
    print(Fore.YELLOW + f"\n  📋 {len(missing)} reading(s) not available from Fitbit — please enter manually:")
    for param in missing:
        label, unit, lo, hi = WEARABLE_PROMPTS[param]
        while True:
            try:
                raw = input(Fore.WHITE + f"   {label} ({unit}) [{lo}–{hi}]: ").strip()
                if raw == "":
                    nr_lo, nr_hi = NORMAL_RANGES.get(param, (lo, hi))
                    data[param] = round((nr_lo + nr_hi) / 2, 1)
                    print(Fore.YELLOW + f"   ↳ Skipped — default {data[param]} {unit}")
                    break
                val = float(raw.replace(",", "."))
                if lo <= val <= hi:
                    data[param] = round(val, 1)
                    break
                else:
                    print(Fore.RED + f"   ❌ Must be {lo}–{hi}")
            except ValueError:
                print(Fore.RED + "   ❌ Please enter a number.")
    return data

def _read_wearable_from_csv(filepath: str):
    import csv as _csv
    rows = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except FileNotFoundError:
        return None
    except Exception as e:
        print(Fore.YELLOW + f"  ⚠️  Could not read {filepath}: {e}")
        return None
    if not rows:
        return None
    raw = rows[-1]
    ALIASES = {
        "heart_rate":    ["heart_rate","heart rate","pulse","hr"],
        "hrv":           ["hrv","heart rate variability","heartratevariability"],
        "spo2":          ["spo2","oxygen","blood oxygen","o2","sp02"],
        "systolic_bp":   ["systolic_bp","systolic","sbp","sys"],
        "diastolic_bp":  ["diastolic_bp","diastolic","dbp","dia"],
        "sleep_hours":   ["sleep_hours","sleep hours","sleep","sleepduration"],
        "sleep_quality": ["sleep_quality","sleep quality","sleepquality"],
        "steps":         ["steps","step count","stepcount","daily_steps"],
        "stress_index":  ["stress_index","stress","stress level","stresslevel"],
    }
    parsed    = {}
    raw_lower = {k.strip().lower().replace(" ","_"): v for k, v in raw.items()}
    for param, aliases in ALIASES.items():
        for alias in aliases:
            key = alias.replace(" ","_")
            if key in raw_lower:
                try:
                    val = float(str(raw_lower[key]).replace(",","."))
                    lo, hi = WEARABLE_PROMPTS[param][2], WEARABLE_PROMPTS[param][3]
                    if lo <= val <= hi:
                        parsed[param] = round(val, 1)
                except (ValueError, TypeError):
                    pass
                break
    return parsed if len(parsed) >= 4 else None


def get_wearable_data() -> dict:
    """
    Priority order:
      1. Fitbit API (if credentials set and user opts in)
      2. CSV file
      3. Manual entry

    Missing fields from Fitbit are always filled by manual prompt.
    """
    divider("WEARABLE DATA SOURCE")
    print(Fore.WHITE + "  [1] 🏃 Sync from Fitbit API  (requires Fitbit setup)")
    print(Fore.WHITE + "  [2] 📄 Load from CSV file    (wearable_data.csv)")
    print(Fore.WHITE + "  [3] ⌨️  Enter manually")
    print()

    while True:
        choice = input(Fore.WHITE + "  Select source (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            break
        print(Fore.RED + "  ❌ Please enter 1, 2 or 3.")

    data = {}

    if choice == "1":
        # ── Fitbit API ────────────────────────────────────────────────────
        fitbit_data = fitbit_api.get_fitbit_data()
        if fitbit_data:
            data = fitbit_data
            print(Fore.GREEN + "\n  ✅ Fitbit data loaded:")
            units = {"heart_rate":"bpm","hrv":"ms","spo2":"%",
                     "systolic_bp":"mmHg","diastolic_bp":"mmHg",
                     "sleep_hours":"hrs","sleep_quality":"%",
                     "steps":"steps","stress_index":"/100"}
            for k, v in data.items():
                lo, hi = NORMAL_RANGES.get(k, (0, 1))
                c = Fore.GREEN if lo <= v <= hi else Fore.YELLOW
                print(c + f"     {k:<22}: {v} {units.get(k,'')}")
        else:
            print(Fore.YELLOW + "  ⚠️  Fitbit unavailable — switching to manual entry.")
        # Fill any missing fields manually
        data = _prompt_missing_wearable(data)

    elif choice == "2":
        # ── CSV ───────────────────────────────────────────────────────────
        csv_data = _read_wearable_from_csv(WEARABLE_CSV)
        if csv_data:
            data = csv_data
            print(Fore.GREEN + f"\n  ✅ Loaded {len(csv_data)} fields from {WEARABLE_CSV}")
        else:
            print(Fore.YELLOW + f"  ⚠️  No valid CSV found at '{WEARABLE_CSV}'.")
        data = _prompt_missing_wearable(data)

    else:
        # ── Full manual entry ─────────────────────────────────────────────
        print(Fore.CYAN + "\n  📋 Enter all wearable readings manually:")
        print(Fore.CYAN + "     (Press Enter to skip — default will be used)\n")
        for param, (label, unit, lo, hi) in WEARABLE_PROMPTS.items():
            while True:
                try:
                    raw = input(Fore.WHITE + f"   {label} ({unit}) [{lo}–{hi}]: ").strip()
                    if raw == "":
                        nr_lo, nr_hi = NORMAL_RANGES.get(param, (lo, hi))
                        data[param] = round((nr_lo + nr_hi) / 2, 1)
                        print(Fore.YELLOW + f"   ↳ Default {data[param]} {unit}")
                        break
                    val = float(raw.replace(",", "."))
                    if lo <= val <= hi:
                        data[param] = round(val, 1)
                        break
                    else:
                        print(Fore.RED + f"   ❌ Must be {lo}–{hi}")
                except ValueError:
                    print(Fore.RED + "   ❌ Please enter a number.")

    return data


def build_feature_vector(lab: dict, wear: dict) -> np.ndarray:
    combined = {**lab, **wear}
    row = []
    for feat in ALL_FEATURES:
        if feat in combined:
            row.append(combined[feat])
        else:
            lo, hi = NORMAL_RANGES.get(feat, (0, 1))
            row.append((lo + hi) / 2)
    return np.array(row, dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
#  3. SYNTHETIC TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
def generate_training_data(n: int = 2000):
    X, y = [], []
    for _ in range(n):
        s, risk = {}, 0
        s["hemoglobin"]      = round(random.uniform(6.0,  18.0), 1)
        s["wbc"]             = round(random.uniform(1.5,  15.0), 2)
        s["platelets"]       = round(random.uniform(30,   600),  0)
        s["rbc"]             = round(random.uniform(2.5,  7.0),  2)
        s["hematocrit"]      = round(random.uniform(20,   60),   1)
        s["mch"]             = round(random.uniform(15,   45),   1)
        s["mcv"]             = round(random.uniform(60,   115),  1)
        s["mchc"]            = round(random.uniform(22,   40),   1)
        s["rdw"]             = round(random.uniform(9,    22),   1)
        s["neutrophils"]     = round(random.uniform(500,  12000),0)
        s["lymphocytes"]     = round(random.uniform(300,  6000), 0)
        s["monocytes"]       = round(random.uniform(100,  2000), 0)
        s["eosinophils"]     = round(random.uniform(0,    2000), 0)
        s["ldl"]             = round(random.uniform(50,   300),  1)
        s["hdl"]             = round(random.uniform(20,   80),   1)
        s["triglycerides"]   = round(random.uniform(50,   500),  1)
        s["fasting_glucose"] = round(random.uniform(60,   300),  1)
        s["hba1c"]           = round(random.uniform(4.0,  12.0), 1)
        s["creatinine"]      = round(random.uniform(0.4,  5.0),  2)
        s["urea"]            = round(random.uniform(5,    80),   1)
        s["sgot"]            = round(random.uniform(10,   150),  1)
        s["sgpt"]            = round(random.uniform(5,    200),  1)
        s["tsh"]             = round(random.uniform(0.1,  10.0), 2)
        s["heart_rate"]      = random.randint(45,  135)
        s["hrv"]             = round(random.uniform(5, 90), 1)
        s["spo2"]            = random.randint(85,  100)
        s["systolic_bp"]     = random.randint(85,  185)
        s["diastolic_bp"]    = random.randint(55,  120)
        s["sleep_hours"]     = round(random.uniform(2.5, 10.0), 1)
        s["sleep_quality"]   = random.randint(10,  100)
        s["steps"]           = random.randint(300, 16000)
        s["stress_index"]    = random.randint(5,   95)

        if s["ldl"]             > 160: risk += 2
        if s["ldl"]             > 200: risk += 2
        if s["hba1c"]           > 6.5: risk += 2
        if s["hba1c"]           > 8.0: risk += 2
        if s["fasting_glucose"] > 126: risk += 2
        if s["systolic_bp"]     > 140: risk += 2
        if s["systolic_bp"]     > 160: risk += 2
        if s["spo2"]            <  95: risk += 3
        if s["spo2"]            <  90: risk += 3
        if s["heart_rate"]      > 100: risk += 1
        if s["hrv"]             <  20: risk += 2
        if s["creatinine"]      >  2.0: risk += 2
        if s["sgpt"]            >  80: risk += 1
        if s["sleep_hours"]     <   5: risk += 1
        if s["stress_index"]    >  70: risk += 1
        if s["hdl"]             <  35: risk += 1
        if s["triglycerides"]   > 200: risk += 1
        if s["hemoglobin"]      <   9: risk += 1
        if s["wbc"]             > 11.0: risk += 1
        if s["wbc"]             <  3.5: risk += 1
        if s["platelets"]       < 100: risk += 1

        label = 0 if risk <= 2 else (1 if risk <= 6 else (2 if risk <= 11 else 3))
        X.append([s[f] for f in ALL_FEATURES])
        y.append(label)
    return np.array(X), np.array(y)

# ─────────────────────────────────────────────────────────────────────────────
#  4. HYBRID ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────────────────────
class HybridEnsemblePredictor:
    def __init__(self):
        self.rf      = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
        self.xgb_clf = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05,
                                          eval_metric="mlogloss", random_state=42, verbosity=0)
        self.meta    = LogisticRegression(max_iter=500, random_state=42)
        self.scaler  = StandardScaler()
        self.shap_ex = None

    def train(self, X, y):
        print(Fore.CYAN + "🤖 Training ensemble models on synthetic clinical data...")
        Xs = self.scaler.fit_transform(X)
        self.rf.fit(Xs, y)
        self.xgb_clf.fit(Xs, y)
        meta_X = np.hstack([self.rf.predict_proba(Xs), self.xgb_clf.predict_proba(Xs)])
        self.meta.fit(meta_X, y)
        self.shap_ex = shap.TreeExplainer(self.rf)
        print(Fore.GREEN + "   ✅ Ensemble training complete!\n")

    def predict(self, X):
        Xs     = self.scaler.transform(X)
        rf_p   = self.rf.predict_proba(Xs)
        xgb_p  = self.xgb_clf.predict_proba(Xs)
        meta_X = np.hstack([rf_p, xgb_p])
        idx    = int(self.meta.predict(meta_X)[0])
        proba  = self.meta.predict_proba(meta_X)[0]
        score  = round(float(np.dot(proba, [15, 45, 70, 92])), 1)
        return idx, score

    def explain(self, X, top_n=8):
        Xs        = self.scaler.transform(X)
        shap_vals = self.shap_ex.shap_values(Xs)
        if isinstance(shap_vals, list):
            abs_imp = np.max(np.abs(np.array(shap_vals))[:, 0, :], axis=0)
        else:
            abs_imp = np.max(np.abs(shap_vals[0]), axis=1) if shap_vals.ndim == 3 else np.abs(shap_vals[0])
        ranked = sorted(zip(ALL_FEATURES, abs_imp), key=lambda x: x[1], reverse=True)[:top_n]
        results = []
        for feat, imp in ranked:
            lo, hi = NORMAL_RANGES.get(feat, (0, 1))
            val    = float(X[0][ALL_FEATURES.index(feat)])
            status = "⚠️  ABNORMAL" if not (lo <= val <= hi) else "✅ Normal"
            results.append({"feature": feat, "value": round(val, 2),
                            "importance": round(float(imp), 4),
                            "status": status, "normal": f"{lo} – {hi}"})
        return results

# ─────────────────────────────────────────────────────────────────────────────
#  5. LONGITUDINAL TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class LongitudinalTracker:
    def __init__(self, user_id: str):
        self.filepath = user_file(user_id, "health_history.json")
        self.history  = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.history, f, indent=2)

    def log(self, risk_score, risk_label, lab_data, wearable_data, disease_json=None):
        self.history.append({
            "timestamp":  datetime.now().isoformat(),
            "risk_score": risk_score,
            "risk_label": risk_label,
            "lab":        lab_data,
            "wearable":   wearable_data,
            "diseases":   disease_json or [],
        })
        self._save()

    def trend_summary(self) -> dict:
        n = len(self.history)
        if n < 2:
            return {"text": "  (First session recorded — trends appear from visit 2 onwards.)",
                    "sessions": n, "avg": 0, "delta": 0, "direction": "STABLE", "anomalies": []}
        recent = self.history[-5:]
        scores = [e["risk_score"] for e in recent]
        avg    = round(sum(scores) / len(scores), 1)
        delta  = round(scores[-1] - scores[0], 1)
        direction = "WORSENING" if delta > 5 else ("IMPROVING" if delta < -5 else "STABLE")
        arrow     = "📈" if direction == "WORSENING" else ("📉" if direction == "IMPROVING" else "➡️")
        anomaly: dict = {}
        for e in recent:
            for feat, val in {**e.get("lab",{}), **e.get("wearable",{})}.items():
                lo, hi = NORMAL_RANGES.get(feat, (None, None))
                if lo is not None and not (lo <= val <= hi):
                    anomaly[feat] = anomaly.get(feat, 0) + 1
        persist = [f for f, c in anomaly.items() if c >= 2]
        lines = [
            f"  Sessions on record : {n}",
            f"  Avg risk (last 5)  : {avg} / 100",
            f"  Trend              : {arrow} {direction}  (Δ {delta:+.1f})",
        ]
        if persist:
            lines.append(f"  ⚠️  Recurrent anomalies : {', '.join(persist)}")
        return {"text": "\n".join(lines), "sessions": n, "avg": avg,
                "delta": delta, "direction": direction, "anomalies": persist}

# ─────────────────────────────────────────────────────────────────────────────
#  6. DASHBOARD EXPORT — now includes disease predictions
# ─────────────────────────────────────────────────────────────────────────────
def export_report_json(user_id, user_name, risk_label, risk_score,
                       lab_data, wearable_data, explanations,
                       trend_dict, disease_json):
    icons = {
        "heart_rate":"❤️","hrv":"📊","spo2":"🫁","systolic_bp":"🩺","diastolic_bp":"🩺",
        "sleep_hours":"😴","sleep_quality":"⭐","steps":"👟","stress_index":"😤",
    }
    wearable_list = [
        {"key": k, "label": k.replace("_"," ").title(), "icon": icons.get(k,"📌"),
         "value": v, "ok": NORMAL_RANGES.get(k,(0,1))[0] <= v <= NORMAL_RANGES.get(k,(0,1))[1]}
        for k, v in wearable_data.items()
    ]
    lab_list = [
        {"name": k.replace("_"," ").title(), "value": str(v),
         "range": f"{NORMAL_RANGES.get(k,(0,1))[0]} – {NORMAL_RANGES.get(k,(0,1))[1]}",
         "abnormal": not (NORMAL_RANGES.get(k,(0,1))[0] <= v <= NORMAL_RANGES.get(k,(0,1))[1])}
        for k, v in lab_data.items()
    ]
    recs_map = {
        "Low":      [{"text":"Great! Maintain your current healthy habits.","color":"#0a7a57"},
                     {"text":"Stay hydrated and exercise regularly.","color":"#0a7a57"},
                     {"text":"Next check-up in 6 months.","color":"#0a7a57"}],
        "Moderate": [{"text":"Follow a low-sodium, low-cholesterol diet.","color":"#e67e22"},
                     {"text":"Walk at least 30 minutes every day.","color":"#e67e22"},
                     {"text":"Aim for 7–8 hours of quality sleep.","color":"#e67e22"},
                     {"text":"See a doctor within 4 weeks.","color":"#e67e22"}],
        "High":     [{"text":"Book a physician consultation soon.","color":"#c04a10"},
                     {"text":"Monitor BP and blood glucose daily.","color":"#c04a10"},
                     {"text":"Avoid smoking, alcohol, and ultra-processed food.","color":"#c04a10"},
                     {"text":"Do NOT self-medicate — seek a proper diagnosis.","color":"#c04a10"}],
        "Critical": [{"text":"SEEK IMMEDIATE MEDICAL ATTENTION.","color":"#c0392b"},
                     {"text":"National Health Helpline: 104","color":"#c0392b"},
                     {"text":"iCall: Dr. Paranthaman A 7397705986","color":"#e67e22"},
                     {"text":"Visit your nearest emergency unit NOW.","color":"#c0392b"}],
    }
    risk_colors = {"Low":"#0a7a57","Moderate":"#e67e22","High":"#c04a10","Critical":"#c0392b"}
    risk_grad   = {
        "Low":      "linear-gradient(90deg,#0a7a57,#0fa878)",
        "Moderate": "linear-gradient(90deg,#e67e22,#f39c12)",
        "High":     "linear-gradient(90deg,#c04a10,#e74c3c)",
        "Critical": "linear-gradient(90deg,#c0392b,#e74c3c)",
    }
    payload = {
        "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "patient_id":   user_id,
        "patient_name": user_name,
        "riskLabel":    risk_label,
        "riskScore":    risk_score,
        "riskColor":    risk_colors.get(risk_label, "#888"),
        "riskBarGrad":  risk_grad.get(risk_label, ""),
        "trend": {
            "direction": trend_dict["direction"],
            "delta":     f"{trend_dict['delta']:+.1f}",
            "sessions":  trend_dict["sessions"],
            "avg":       trend_dict["avg"],
        },
        "anomalies":  trend_dict["anomalies"],
        "wearable":   wearable_list,
        "shap":       [{"feature": e["feature"], "value": e["value"],
                        "normal": e["normal"], "abnormal": "ABNORMAL" in e["status"]}
                       for e in explanations],
        "lab":      lab_list,
        "recs":     recs_map.get(risk_label, []),
        "diseases": disease_json,   # ← NEW: proactive disease forecast
    }
    report_path = user_file(user_id, "report_data.json")
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open("report_data.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(Fore.GREEN + f"📊 Report saved → {report_path}")


def open_dashboard():
    path = os.path.abspath(DASHBOARD_HTML)
    if not os.path.exists(path):
        print(Fore.YELLOW + f"  ⚠️  Dashboard not found: {path}")
        return
    webbrowser.open(f"file:///{path.replace(os.sep, '/')}")
    print(Fore.GREEN + f"🌐 Dashboard opened → {DASHBOARD_HTML}")

# ─────────────────────────────────────────────────────────────────────────────
#  7. LLM CHAT — disease-aware context
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = """You are a medical assistant AI. Be concise and warm.
Do NOT prescribe medication. Advise a doctor visit for High/Critical risk.

OVERALL RISK: {risk_label} ({risk_score}/100)
ABNORMAL VALUES: {abnormal_only}
PREDICTED DISEASES: {disease_summary}
RECENT HISTORY: {context}

Patient: {question}
Assistant:"""

llm   = OllamaLLM(model="llama3.2:1b")
chain = ChatPromptTemplate.from_template(TEMPLATE) | llm


def ask_llm(risk_score, risk_label, shap_items, question, history, disease_json) -> str:
    abnormal = [f"{e['feature']}={e['value']} (normal {e['normal']})"
                for e in shap_items if "ABNORMAL" in e["status"]]
    abnormal_str = ", ".join(abnormal) if abnormal else "None"

    # Include top 3 diseases in LLM context
    top_diseases = [f"{d['disease']} ({d['risk_tier']}, {d['risk_pct']:.0f}%)"
                    for d in disease_json[:3]] if disease_json else ["None flagged"]
    disease_str  = "; ".join(top_diseases)

    ctx_str = "\n".join(list(history)[-2:]) if history else "None"
    result  = chain.invoke({
        "risk_label":      risk_label,
        "risk_score":      risk_score,
        "abnormal_only":   abnormal_str,
        "disease_summary": disease_str,
        "context":         ctx_str,
        "question":        question,
    })
    return result.content.strip() if hasattr(result, "content") else str(result)

# ─────────────────────────────────────────────────────────────────────────────
#  8. DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def divider(title=""):
    w = 68
    if title:
        pad = (w - len(title) - 2) // 2
        print(Fore.CYAN + "─"*pad + f" {title} " + "─"*pad)
    else:
        print(Fore.CYAN + "─"*w)

def print_risk_banner(score, risk_label, idx):
    c = RISK_COLORS[idx]
    bar = "█"*int(score/5) + "░"*(20-int(score/5))
    divider("RISK ASSESSMENT")
    print(c + f"  Level : {risk_label.upper()}")
    print(c + f"  Score : {score:>5.1f} / 100")
    print(c + f"  [{bar}]")
    divider()

def print_lab_table(lab_data: dict):
    divider("EXTRACTED LAB PARAMETERS")
    if not lab_data:
        print(Fore.YELLOW + "  ⚠️  No standard lab values detected in the report.")
        print(Fore.YELLOW + "  ℹ️  Prediction will use wearable data only.")
        divider(); return
    print(f"  {'Parameter':<25} {'Value':>10}   {'Normal Range':<18} Status")
    print("  " + "─"*65)
    for k, v in sorted(lab_data.items()):
        lo, hi = NORMAL_RANGES.get(k, (None, None))
        if lo is not None:
            ok = lo <= v <= hi
            c  = Fore.GREEN if ok else Fore.RED
            print(c + f"  {k:<25} {str(v):>10}   {lo} – {hi:<12} {'✅ Normal' if ok else '⚠️  ABNORMAL'}")
        else:
            print(Fore.WHITE + f"  {k:<25} {str(v):>10}   {'—':<18} —")
    divider()

def print_shap_table(explanations):
    divider("EXPLAINABLE AI — TOP RISK DRIVERS (SHAP)")
    print(f"  {'Feature':<22} {'Value':>8}   {'Normal Range':<16} Status")
    print("  " + "─"*62)
    for e in explanations:
        c = Fore.RED if "ABNORMAL" in e["status"] else Fore.GREEN
        print(c + f"  {e['feature']:<22} {str(e['value']):>8}   {e['normal']:<16} {e['status']}")
    divider()

def print_wearable(wearable_data):
    units = {"heart_rate":"bpm","hrv":"ms","spo2":"%","systolic_bp":"mmHg",
             "diastolic_bp":"mmHg","sleep_hours":"hrs","sleep_quality":"%",
             "steps":"steps","stress_index":"/100"}
    divider("WEARABLE SENSOR DATA")
    for k, v in wearable_data.items():
        lo, hi = NORMAL_RANGES.get(k, (None, None))
        ok = (lo <= v <= hi) if lo is not None else True
        print((Fore.GREEN if ok else Fore.YELLOW) + f"  {k:<22}: {v} {units.get(k,'')}")
    divider()

def print_recommendations(risk_label):
    recs = {
        "Low":      ["✅ Great! Maintain your current healthy habits.",
                     "💧 Stay hydrated and exercise regularly.",
                     "📅 Next check-up in 6 months."],
        "Moderate": ["🥗 Follow a low-sodium, low-cholesterol diet.",
                     "🚶 Walk at least 30 minutes every day.",
                     "😴 Aim for 7–8 hours of quality sleep.",
                     "📅 See a doctor within 4 weeks."],
        "High":     ["🏥 Book a physician consultation soon.",
                     "📉 Monitor BP and blood glucose daily.",
                     "🚫 Avoid smoking, alcohol, and ultra-processed food.",
                     "💊 Do NOT self-medicate — seek a proper diagnosis."],
        "Critical": ["🚨 SEEK IMMEDIATE MEDICAL ATTENTION.",
                     "📞 National Health Helpline: 104",
                     "📞 iCall: Dr. Paranthaman A 7397705986",
                     "🏥 Visit your nearest emergency unit NOW."],
    }
    divider("PERSONALISED RECOMMENDATIONS")
    for rec in recs.get(risk_label, []):
        print(Fore.WHITE + f"  {rec}")
    divider()

def print_disease_predictions(disease_results):
    """★ NEW — print proactive disease prediction section."""
    divider("PROACTIVE DISEASE RISK FORECAST")
    if not disease_results:
        print(Fore.GREEN + "  ✅ No significant disease risk flags detected.")
        divider(); return

    tier_icon  = {"Low":"🟢","Moderate":"🟡","High":"🔴","Critical":"🚨"}
    tier_color = {"Low": Fore.GREEN,"Moderate": Fore.YELLOW,
                  "High": Fore.RED,"Critical": Fore.MAGENTA}

    for r in disease_results:
        c = tier_color.get(r.risk_tier, Fore.WHITE)
        print(c + f"\n  {r.icon}  {r.disease}")
        print(c + f"      Risk: {tier_icon.get(r.risk_tier,'')} {r.risk_tier}  "
                  f"({r.risk_pct:.0f}/100)")
        if r.key_drivers:
            print(Fore.WHITE + "      Key factors:")
            for d in r.key_drivers[:3]:
                print(Fore.WHITE + f"        • {d}")
        print(Fore.CYAN + f"      💡 {r.advice}")
    divider()

def print_previous_visits(history: list, user_name: str):
    if not history:
        return
    divider(f"VISIT HISTORY — {user_name.upper()}")
    print(f"  {'#':<4} {'Date':<20} {'Risk Level':<12} {'Score':>6}")
    print("  " + "─"*45)
    for i, entry in enumerate(history[-5:], 1):
        ts    = entry.get("timestamp", "")[:16].replace("T", " ")
        label = entry.get("risk_label", "—")
        score = entry.get("risk_score", 0)
        idx   = RISK_LABELS.index(label) if label in RISK_LABELS else 0
        print(RISK_COLORS[idx] + f"  {i:<4} {ts:<20} {label:<12} {score:>6.1f}")
    divider()

# ─────────────────────────────────────────────────────────────────────────────
#  9. MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────
def load_or_train_predictor() -> "HybridEnsemblePredictor":
    if os.path.exists(GLOBAL_MODEL):
        print(Fore.CYAN + f"📦 Loading cached model from {GLOBAL_MODEL}...")
        with open(GLOBAL_MODEL, "rb") as f:
            predictor = pickle.load(f)
        print(Fore.GREEN + "   ✅ Model loaded instantly!\n")
        return predictor
    X_tr, y_tr = generate_training_data(n=2000)
    predictor  = HybridEnsemblePredictor()
    predictor.train(X_tr, y_tr)
    with open(GLOBAL_MODEL, "wb") as f:
        pickle.dump(predictor, f)
    print(Fore.CYAN + f"💾 Model cached → {GLOBAL_MODEL}\n")
    return predictor

# ─────────────────────────────────────────────────────────────────────────────
#  10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(Style.BRIGHT + Fore.CYAN + """
╔══════════════════════════════════════════════════════════════════╗
║  MULTIMODAL EXPLAINABLE AI — PREDICTIVE HEALTH INTELLIGENCE     ║
║  PSNA College of Engineering and Technology  |  AI & DS Dept   ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    os.makedirs(USERS_ROOT, exist_ok=True)
    user_id, user_name, is_new_user = select_user()

    predictor = load_or_train_predictor()
    tracker   = LongitudinalTracker(user_id)

    if tracker.history:
        print_previous_visits(tracker.history, user_name)

    # ── Lab Report (upload or reuse) ────────────────────────────────────────
    lab_data, report_source = get_lab_report(user_id, is_new_user)
    wearable_data = get_wearable_data()

    print_lab_table(lab_data)

    # ── ML Prediction ────────────────────────────────────────────────────────
    fv           = build_feature_vector(lab_data, wearable_data).reshape(1, -1)
    idx, score   = predictor.predict(fv)
    risk_label   = RISK_LABELS[idx]
    explanations = predictor.explain(fv)

    print_wearable(wearable_data)
    print_risk_banner(score, risk_label, idx)
    print_shap_table(explanations)

    # ── ★ PROACTIVE DISEASE PREDICTION ★ ────────────────────────────────────
    disease_results = predict_diseases(lab_data, wearable_data, threshold=20.0)
    disease_json    = diseases_to_json(disease_results)
    print_disease_predictions(disease_results)

    # ── General recommendations ──────────────────────────────────────────────
    print_recommendations(risk_label)

    # ── Trend ────────────────────────────────────────────────────────────────
    tracker.log(score, risk_label, lab_data, wearable_data, disease_json)
    trend_dict = tracker.trend_summary()
    divider("LONGITUDINAL HEALTH TREND")
    print(Fore.WHITE + trend_dict["text"])
    divider()

    # ── Dashboard ────────────────────────────────────────────────────────────
    export_report_json(user_id, user_name, risk_label, score, lab_data,
                       wearable_data, explanations, trend_dict, disease_json)
    open_dashboard()

    # ── LLM Chat ─────────────────────────────────────────────────────────────
    print(Fore.CYAN + f"\n💬 Hi {user_name}! Ask anything about your report — type 'bye' to exit.\n")
    history: deque = deque(maxlen=6)

    while True:
        try:
            user_input = input(Fore.WHITE + "You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("bye", "exit", "quit"):
            print(Fore.GREEN + "Bot: Stay healthy! Take care. 💙")
            break
        if not user_input:
            continue
        print(Fore.YELLOW + "Bot: Thinking... ⏳")
        response = ask_llm(score, risk_label, explanations,
                           user_input, history, disease_json)
        history.append(f"Patient: {user_input}\nAssistant: {response}")
        print(Fore.GREEN + f"Bot: {response}\n")


if __name__ == "__main__":
    main()
