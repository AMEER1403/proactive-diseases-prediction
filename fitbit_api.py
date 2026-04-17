"""
fitbit_api.py
─────────────────────────────────────────────────────────────────────────────
Fitbit OAuth 2.0 integration for the Health Intelligence System.

HOW TO SET UP FITBIT API:
  1. Go to https://dev.fitbit.com/apps/new
  2. Create an app (OAuth 2.0 Application Type: Personal)
  3. Set Redirect URI to: http://localhost:8765/callback
  4. Copy Client ID and Client Secret into config below

FLOW:
  get_fitbit_data()
    → tries to load saved token
    → if no token / expired → runs OAuth2 browser flow
    → fetches today's heart rate, SpO2, sleep, steps, stress proxy
    → returns dict compatible with wearable_data format
─────────────────────────────────────────────────────────────────────────────
"""

import os, json, time, webbrowser, threading
from datetime import date
from urllib.parse import urlencode, urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

from colorama import Fore, Style

# ── CONFIGURE YOUR FITBIT APP CREDENTIALS HERE ───────────────────────────────
FITBIT_CLIENT_ID     = os.environ.get("FITBIT_CLIENT_ID",     "YOUR_CLIENT_ID")
FITBIT_CLIENT_SECRET = os.environ.get("FITBIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
FITBIT_REDIRECT_URI  = "http://localhost:8765/callback"
FITBIT_SCOPES        = "heartrate oxygen_saturation sleep activity"
TOKEN_FILE           = "fitbit_token.json"   # one per machine (not per-user, Fitbit is personal)

# Fitbit API base
API_BASE = "https://api.fitbit.com/1/user/-"

# ─────────────────────────────────────────────────────────────────────────────
#  TOKEN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_token(token: dict):
    token["saved_at"] = time.time()
    with open(TOKEN_FILE, "w") as f:
        json.dump(token, f, indent=2)


def _load_token() -> dict | None:
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE) as f:
        return json.load(f)


def _is_token_valid(token: dict) -> bool:
    saved_at    = token.get("saved_at", 0)
    expires_in  = token.get("expires_in", 0)
    return (time.time() - saved_at) < (expires_in - 60)


def _refresh_token(token: dict) -> dict | None:
    """Try to refresh the access token using the refresh token."""
    import urllib.request, base64
    creds  = base64.b64encode(f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()).decode()
    data   = urlencode({
        "grant_type":    "refresh_token",
        "refresh_token": token["refresh_token"],
    }).encode()
    req    = urllib.request.Request(
        "https://api.fitbit.com/oauth2/token",
        data    = data,
        headers = {
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            new_token = json.loads(resp.read())
            _save_token(new_token)
            return new_token
    except Exception as e:
        print(Fore.RED + f"  ❌ Token refresh failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  OAUTH 2.0 BROWSER FLOW
# ─────────────────────────────────────────────────────────────────────────────

_auth_code: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _auth_code
        qs = parse_qs(urlparse(self.path).query)
        _auth_code = qs.get("code", [None])[0]
        body = b"<h2>Authorization complete! You can close this tab.</h2>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress server logs


def _run_browser_flow() -> dict | None:
    """Opens Fitbit consent page, captures auth code, exchanges for token."""
    global _auth_code
    import urllib.request, base64

    # Build authorization URL
    params = {
        "response_type": "code",
        "client_id":     FITBIT_CLIENT_ID,
        "redirect_uri":  FITBIT_REDIRECT_URI,
        "scope":         FITBIT_SCOPES,
        "expires_in":    "604800",   # 1 week
    }
    auth_url = "https://www.fitbit.com/oauth2/authorize?" + urlencode(params)

    print(Fore.CYAN + "\n🔗 Fitbit Authorization Required")
    print(Fore.WHITE + "   Opening your browser for Fitbit login...")
    print(Fore.YELLOW + f"   If it doesn't open, visit:\n   {auth_url}\n")

    # Start local callback server in a thread
    server = HTTPServer(("localhost", 8765), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    webbrowser.open(auth_url)
    thread.join(timeout=120)
    server.server_close()

    if not _auth_code:
        print(Fore.RED + "  ❌ No authorization code received (timed out).")
        return None

    # Exchange code for token
    creds = base64.b64encode(f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()).decode()
    data  = urlencode({
        "grant_type":   "authorization_code",
        "code":         _auth_code,
        "redirect_uri": FITBIT_REDIRECT_URI,
    }).encode()
    req = urllib.request.Request(
        "https://api.fitbit.com/oauth2/token",
        data    = data,
        headers = {
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            token = json.loads(resp.read())
            _save_token(token)
            print(Fore.GREEN + "  ✅ Fitbit authorization successful!\n")
            return token
    except Exception as e:
        print(Fore.RED + f"  ❌ Token exchange failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  FITBIT API CALLS
# ─────────────────────────────────────────────────────────────────────────────

def _api_get(endpoint: str, access_token: str) -> dict | None:
    import urllib.request
    url = f"{API_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(Fore.YELLOW + f"  ⚠️  API call failed ({endpoint}): {e}")
        return None


def _fetch_today(access_token: str) -> dict:
    today     = date.today().isoformat()
    wearable  = {}

    # ── Heart Rate ──────────────────────────────────────────────────────────
    hr_data = _api_get(f"/activities/heart/date/{today}/1d.json", access_token)
    if hr_data:
        try:
            resting_hr = hr_data["activities-heart"][0]["value"].get("restingHeartRate")
            if resting_hr:
                wearable["heart_rate"] = float(resting_hr)
        except (KeyError, IndexError):
            pass

    # ── HRV ─────────────────────────────────────────────────────────────────
    hrv_data = _api_get(f"/hrv/date/{today}.json", access_token)
    if hrv_data:
        try:
            hrv_val = hrv_data["hrv"][0]["value"]["dailyRmssd"]
            wearable["hrv"] = round(float(hrv_val), 1)
        except (KeyError, IndexError):
            pass

    # ── SpO2 ─────────────────────────────────────────────────────────────────
    spo2_data = _api_get(f"/spo2/date/{today}.json", access_token)
    if spo2_data:
        try:
            wearable["spo2"] = round(float(spo2_data["value"]["avg"]), 1)
        except (KeyError, TypeError):
            pass

    # ── Sleep ────────────────────────────────────────────────────────────────
    sleep_data = _api_get(f"/sleep/date/{today}.json", access_token)
    if sleep_data:
        try:
            mins = sleep_data["summary"]["totalMinutesAsleep"]
            wearable["sleep_hours"]   = round(mins / 60, 1)
            eff = sleep_data["summary"].get("efficiency", None)
            if eff is not None:
                wearable["sleep_quality"] = float(eff)
        except (KeyError, TypeError):
            pass

    # ── Steps ────────────────────────────────────────────────────────────────
    steps_data = _api_get(f"/activities/date/{today}.json", access_token)
    if steps_data:
        try:
            wearable["steps"] = int(steps_data["summary"]["steps"])
        except (KeyError, TypeError):
            pass

    # ── Stress proxy: resting HR deviation ──────────────────────────────────
    # Fitbit doesn't expose stress score via free API.
    # We approximate: high resting HR → higher stress index.
    if "heart_rate" in wearable:
        hr = wearable["heart_rate"]
        stress = max(0, min(100, int((hr - 60) * 2.5)))
        wearable["stress_index"] = stress

    # BP is not available from Fitbit (no sensor) — will be filled manually
    return wearable


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def get_fitbit_data() -> dict | None:
    """
    Returns a dict of wearable values pulled from Fitbit API today,
    or None if unavailable / user skips.

    Missing fields (e.g. BP) will be None — caller should prompt for them.
    """
    if FITBIT_CLIENT_ID == "YOUR_CLIENT_ID":
        print(Fore.YELLOW + "  ℹ️  Fitbit API credentials not configured.")
        print(Fore.YELLOW + "     Set FITBIT_CLIENT_ID and FITBIT_CLIENT_SECRET in fitbit_api.py\n")
        return None

    # Try loading a saved token
    token = _load_token()

    if token and _is_token_valid(token):
        print(Fore.CYAN + "  🔄 Using saved Fitbit token...")
    elif token and token.get("refresh_token"):
        print(Fore.CYAN + "  🔄 Refreshing Fitbit token...")
        token = _refresh_token(token)
    else:
        print(Fore.CYAN + "  🔑 Starting Fitbit OAuth2 authorization flow...")
        token = _run_browser_flow()

    if not token:
        return None

    print(Fore.CYAN + "  📡 Fetching today's data from Fitbit API...")
    data = _fetch_today(token["access_token"])

    if not data:
        print(Fore.YELLOW + "  ⚠️  No Fitbit data retrieved.")
        return None

    print(Fore.GREEN + f"  ✅ Fitbit synced: {len(data)} metric(s) fetched")
    return data
