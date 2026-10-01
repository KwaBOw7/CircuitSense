"""Gemini access. The AI identifies unknown symbols and explains solver output; it never calculates.
Set GEMINI_API_KEY. Optional: GEMINI_MODEL to force a model (otherwise the newest Flash model your key can use is picked)."""
import os, json, base64, urllib.request, urllib.error
KINDS = {"R", "C", "L", "V", "I", "D", "LED", "UNKNOWN"}
PREFER = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash", "gemini-flash-latest"]
BASE = "https://generativelanguage.googleapis.com/v1beta"
PROMPT = ('This image is a crop of ONE electrical schematic symbol. Return JSON only: '
          '{"kind": one of R,C,L,V,I,D,LED,UNKNOWN, "value": number in SI units or null, '
          '"reversed": true if the left/top terminal is the negative/cathode side}.')
_model = None

def _http(url, body=None):
    req = urllib.request.Request(url, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")

def _pick(key, refresh=False):
    global _model
    if os.getenv("GEMINI_MODEL"): return os.environ["GEMINI_MODEL"]
    if _model and not refresh: return _model
    names = [m["name"].split("/")[-1] for m in _http(f"{BASE}/models?key={key}&pageSize=200").get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    flash = [n for n in names if "flash" in n and not any(w in n for w in ("image", "tts", "live", "audio", "lite", "thinking", "exp"))]
    _model = next((p for p in PREFER if p in names), None) or (sorted(flash)[-1] if flash else (names[0] if names else None))
    if not _model: raise RuntimeError("No Gemini model with generateContent is available for this key")
    return _model

def _generate(parts, as_json):
    key = os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY is not set")
    body = {"contents": [{"parts": parts}]}
    if as_json: body["generationConfig"] = {"response_mime_type": "application/json"}
    try:
        try: d = _http(f"{BASE}/models/{_pick(key)}:generateContent?key={key}", body)
        except RuntimeError as e:
            if "HTTP 404" not in str(e) or os.getenv("GEMINI_MODEL"): raise
            d = _http(f"{BASE}/models/{_pick(key, True)}:generateContent?key={key}", body)   # stale model name: re-discover
        out = d["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(out) if as_json else out
    except RuntimeError as e:
        raise RuntimeError(f"Gemini call failed ({_model}): {e}")
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Gemini returned an unexpected response: {type(e).__name__}")

def generate(text, as_json=False):
    return _generate([{"text": text}], as_json)

def resolve(png: bytes):
    """Identify ONE unrecognised symbol crop -> {'kind','value','reversed'} or None."""
    if not os.getenv("GEMINI_API_KEY"): return None
    try:
        d = _generate([{"text": PROMPT}, {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}}], True)
    except RuntimeError:
        return None
    return d if isinstance(d, dict) and d.get("kind") in KINDS else None


def generate_image(text, png: bytes, as_json=False):
    return _generate([{"text": text}, {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}}], as_json)
