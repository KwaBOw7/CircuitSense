import os, time, collections
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib
from .vision import analyze
from .solver import solve, whatif
from . import qa, ladder_sim, wiring, ladder_explain, inverse, ladder_wiring, acsolve, netlist
import json
from .ladder import analyze_ladder
from .model import Circuit

def _load_env(path=None):
    """Read KEY=VALUE lines from a local .env (git-ignored) without overriding real environment variables. Works with Notepad's BOM."""
    p = pathlib.Path(path) if path else pathlib.Path(__file__).parent.parent / ".env"
    if not p.is_file(): return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1); k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v
_load_env()

app = FastAPI(title="CircuitSense", version="1.1")
MAX_BYTES = 15_000_000
_hits = collections.defaultdict(collections.deque)

def _limit(request: Request, bucket: str, env: str, default: int):
    """Per-IP hourly cap on the expensive endpoints, so a public demo can't burn the Gemini quota or the CPU."""
    n = int(os.getenv(env, default)); ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    if len(_hits) > 5000: _hits.clear()
    q, now = _hits[(bucket, ip)], time.time()
    while q and now - q[0] > 3600: q.popleft()
    if len(q) >= n: raise HTTPException(429, f"Rate limit reached ({n} per hour for this endpoint). Try again later.")
    q.append(now)

async def _image(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "image too large (15 MB max)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "not a readable image")
    return img

@app.get("/health")
def health(): return {"ok": True}

@app.post("/analyze")
async def analyze_ep(request: Request, file: UploadFile, wire_frac: float = 0.06, join_crossings: bool = False, kind: str = "circuit"):
    _limit(request, "analyze", "RATE_LIMIT_ANALYZE", 200)
    img = await _image(file)
    if kind == "ladder": return analyze_ladder(img)
    return analyze(img, wire_frac, join_crossings=join_crossings).to_dict()

@app.post("/solve")   # body: circuit JSON from /analyze with values filled in / corrected
def solve_ep(circuit: dict):
    try: return solve(Circuit.from_dict(circuit))
    except (ValueError, np.linalg.LinAlgError) as e: raise HTTPException(422, str(e))

@app.post("/find")     # body: {"circuit": {...values null for unknowns}, "constraints": [{"kind":"v","of":"R2","value":3}]}
def find_ep(body: dict):
    try: return inverse.find(Circuit.from_dict(body["circuit"]), body["constraints"])
    except (ValueError, KeyError, np.linalg.LinAlgError) as e: raise HTTPException(422, str(e))

@app.post("/ac")       # body: {"circuit": {...}, "freq": 1000}
def ac_ep(body: dict):
    try: return acsolve.ac_solve(Circuit.from_dict(body["circuit"]), float(body["freq"]))
    except (ValueError, KeyError, TypeError) as e: raise HTTPException(422, str(e))

@app.post("/ac/sweep") # body: {"circuit": {...}, "of": "node:2" | "v:R1" | "i:C1", "f0": 10, "f1": 100000, "n": 120}
def ac_sweep_ep(body: dict):
    try: return acsolve.sweep(Circuit.from_dict(body["circuit"]), body["of"], float(body["f0"]), float(body["f1"]), int(body.get("n", 120)))
    except (ValueError, KeyError, TypeError) as e: raise HTTPException(422, str(e))

@app.post("/import")   # multipart file: KiCad SPICE netlist (.cir/.sp) or KiCad .net
async def import_ep(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "file too large")
    try: return netlist.load(data.decode("utf-8", "replace")).to_dict()
    except (ValueError, IndexError, KeyError, StopIteration) as e: raise HTTPException(422, str(e) or "could not parse this netlist")

@app.post("/whatif")   # body: {"circuit": {...}, "changes": {"R2": 10}}
def whatif_ep(body: dict):
    try: return whatif(Circuit.from_dict(body["circuit"]), body["changes"])
    except (ValueError, KeyError, np.linalg.LinAlgError) as e: raise HTTPException(422, str(e))

@app.post("/ask")      # body: {"circuit": {...}, "question": "..."}
def ask_ep(body: dict, request: Request):
    _limit(request, "ai", "RATE_LIMIT_AI", 30)
    try: return qa.ask(body["circuit"], body["question"])
    except RuntimeError as e: raise HTTPException(503 if "not set" in str(e) else 502, str(e))
    except (ValueError, KeyError, np.linalg.LinAlgError) as e: raise HTTPException(422, str(e))

@app.post("/ladder/sim")   # body: {"ladder": {...}, "inputs": {"X0": true}, "state": {...}}
def ladder_ep(body: dict):
    lad = body["ladder"]; ins, outs = ladder_sim.io(lad)
    st = ladder_sim.scan(lad, body.get("state", {}), body.get("inputs", {}))
    return {"state": st, "inputs": ins, "outputs": outs, "equations": ladder_sim.equations(lad)}

@app.post("/ladder/explain")   # multipart: file (the image) + ladder (JSON string from /analyze?kind=ladder)
async def explain_ep(request: Request, file: UploadFile, ladder: str = Form("{}")):     # Form(): otherwise FastAPI reads it as a query param and silently drops it
    _limit(request, "ai", "RATE_LIMIT_AI", 30)
    img = await _image(file)
    return ladder_explain.explain(json.loads(ladder), cv2.imencode(".png", img)[1].tobytes())

@app.post("/ladder/wiring")   # body: {"ladder": {...}, "internal": ["Latch Coil"]}  -> PLC I/O wiring table + SVG
def ladder_wiring_ep(body: dict):
    lad = body["ladder"]
    if not lad.get("elements"): raise HTTPException(422, "no ladder elements to wire")
    return ladder_wiring.generate(lad, body.get("internal", []))

@app.post("/wiring")       # body: {"circuit": {...}, "a": "R1.1", "b": "V1.+"}
def wiring_ep(body: dict):
    return wiring.analyze(Circuit.from_dict(body["circuit"]), body.get("a"), body.get("b"))

SAMPLES = pathlib.Path(__file__).parent.parent / "samples"

@app.get("/samples")
def samples(): return sorted(p.name for p in SAMPLES.iterdir() if p.suffix.lower() in (".png", ".jpg", ".cir", ".net", ".sp"))

@app.get("/samples/{name}")
def sample(name: str):
    p = SAMPLES / pathlib.Path(name).name                      # no path traversal
    if not p.is_file(): raise HTTPException(404, "no such sample")
    return FileResponse(p)

@app.get("/")
def ui(): return FileResponse(pathlib.Path(__file__).parent.parent / "static" / "index.html")

app.mount("/static", StaticFiles(directory=pathlib.Path(__file__).parent.parent / "static"), name="static")
