"""Label reading (RapidOCR, pip-only) + value parsing. Returns nothing if the engine is missing."""
import re
_eng = None
error = None   # set when the OCR engine can't run, so the app can say so instead of failing silently
_U = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}
_DES = re.compile(r"[A-Za-z]{1,2}\d{0,2}")
_LABEL = re.compile(r"[RCLVIBQDE]\d{1,2}|[IVE]|\d+(?:[.,]\d+)?[pnuµmkKMG]?\S{0,2}")

def read_all(img):
    global _eng, error
    try:
        from rapidocr_onnxruntime import RapidOCR
        _eng = _eng or RapidOCR()
        res, _ = _eng(img)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        return []
    out = []
    for b, t, sc in res or []:
        xs, ys = [p[0] for p in b], [p[1] for p in b]
        if float(sc) >= 0.5: out.append((t.strip(), (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))))
    return out

def is_label(t):   # only text that looks like a designator or a value; zigzags read as "W" etc. stay in the image
    return len(t) <= 8 and bool(_LABEL.fullmatch(t.replace(" ", "")))

def split(t, glyphs):
    """-> (value, unit). Omega is often misread as a digit/letter, so if OCR returned as many chars as there are glyphs, the last one is the unit."""
    t = t.replace(" ", "")
    if not t or _DES.fullmatch(t): return None, None
    u = "A" if t.endswith("A") else "V" if t.endswith("V") else "Ω"
    if u != "Ω": t = t[:-1]
    elif len(t) == glyphs and t[-1].isdigit() and len(t) > 1: t = t[:-1]
    m = re.match(r"(\d+(?:[.,]\d+)?)([pnuµmkKMG]?)", t)
    return (float(m.group(1).replace(",", ".")) * _U.get(m.group(2), 1), u) if m else (None, None)

def norm_addr(t):
    """PLC address cleanup: 'YO.' -> 'Y0' (letter O for zero, trailing punctuation)."""
    t = t.replace(" ", "").upper().rstrip(".:/")
    m = re.fullmatch(r"(%?[A-Z])([O0-9.:/]+)", t)
    return m.group(1) + m.group(2).replace("O", "0") if m else t
