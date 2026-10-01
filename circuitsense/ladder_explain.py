"""Explain a ladder diagram in plain language and identify the real-world system it controls.
Structure the parser is sure of (no warnings) is passed as ground truth; otherwise the model reads the image itself."""
import json
from . import ai_fallback, ladder_sim

PROMPT = """You are explaining a PLC ladder-logic diagram to an engineering student. Look at the image.
{grounding}
Return JSON only:
{{"system": short name of the real-world system this most likely controls (e.g. "Tank level control with pump"),
 "confidence": "high"|"medium"|"low",
 "summary": 2-4 sentences: what the whole program does, in plain language,
 "rungs": [{{"rung": number, "explanation": string}}],
 "notes": [strings: assumptions, unreadable parts, why you identified the system this way]}}
Rules: describe only what is visible; if something is unreadable say so; if the identification is a guess give low confidence and say why.
Explain manual/auto selection, latch/unlatch behaviour, and what each output (motor, lamps) means for the process."""

def explain(lad, png=None, gen=None):
    clean = bool(lad and lad.get("elements")) and not lad.get("warnings")
    det = ladder_sim.rung_text(lad) if clean else []
    ground = ("Verified rung logic from a deterministic parser (trust this over your own reading):\n" + "\n".join(det)) if clean else \
             ("Text an OCR engine read from the image (may contain errors): " + json.dumps((lad or {}).get("texts", [])))
    res = {"deterministic": det, "verified": clean, "ai": None, "note": None}
    try:
        p = PROMPT.format(grounding=ground)
        res["ai"] = gen(p, png, True) if gen else (ai_fallback.generate_image(p, png, True) if png else ai_fallback.generate(p, True))
    except RuntimeError as e:
        res["note"] = str(e)
    return res
