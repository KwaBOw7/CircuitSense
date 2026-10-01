# CircuitSense

Upload a picture of an electrical diagram. CircuitSense turns it into a structured model, analyses it with
**deterministic code**, and only uses AI where language is the job: explaining a ladder program and answering questions.

| Diagram | What you get |
|---|---|
| **Circuit schematic** | Parts + connections detected (R, C, L, DC sources, current sources, LED/diode, ground), values read by OCR, full DC solution (node voltages, current, power per part), **what-if** re-solving, nets/continuity/short checks |
| **Transistors & op-amps (DC)** | **BJT** NPN/PNP (Vbe 0.7 V, β, saturation at Vce 0.2 V) with the operating region reported (cutoff / active / saturation); **MOSFET** N/P level-1 square law (cutoff / triode / saturation); **ideal op-amp** (inverting, non-inverting, optional supply rails → clipping) and **VCVS**. Editable in the Parts table (`beta`, `vth`, `kp`, `vpos`/`vneg`) |
| **AC analysis** | Phasor solver for R, C, L, AC sources and **ideal op-amps / VCVS** (active filters): magnitude/phase of every voltage and current, average and reactive power, and a **frequency sweep** with Bode plot, −3 dB cutoff, resonance and bandwidth (diodes and transistors are DC-only; AC refuses them with a clear message) |
| **KiCad netlist import** | Upload a KiCad SPICE netlist (`.cir`) or `.net`: exact connectivity, no image recognition. Values like `4k7`, `100n`, `SIN(0 1 1k)` are understood; **transistors (Q), MOSFETs (M), op-amps (U with +/−/out pins) and VCVS (E)** are imported with their `.model` parameters; anything it can't identify is flagged, never guessed |
| **Ladder logic** | Rungs, contacts (NO/NC), coils (normal/latch/unlatch), addresses; **plain-English rung explanation, system identification**, Boolean equations, live scan simulation |
| **Missing values** | Leave a value blank and give measurements ("2.5 V across R2"); the solver **finds the unknowns**, and says when the measurements can't pin them down (NOT unique) or contradict the circuit |
| **Ladder → wiring** | Generates a **PLC I/O wiring diagram** from the ladder: inputs → field devices (S/B) → DIn, outputs → DOn → loads (H/KM/K) → 0V, with wire numbers, a from/to table, terminal list, and printable SVG |
| **Wiring (analysis)** | Nets, continuity query ("is R1.1 connected to V1.+?"), dangling terminals, shorted sources |

## Quick demo (2 minutes)
Use the **LOAD SAMPLE…** menu in the console bar:
1. `ce_amp.cir` → *Solve*: BJT lands in **active** region, Ic = 1 mA, Vc = 5 V. *What If* → change RB to 100 k and watch it go to **saturation**.
2. `inverting_opamp.net` → *Solve*: Vout = −10 V (gain −Rf/Rin). *AC* tab → sweep.
3. `rc_lowpass.cir` → *AC* tab → sweep `node` at the output: −3 dB corner at 1 kHz on the Bode plot.
4. `divider.net` → shows the *never guess* behaviour: a transistor with a floating pin is flagged, not solved.
5. `sample.png` → *Analyze*, then *Unknowns*: give a measurement and the solver finds the missing values.
6. `ladder.png` (Ladder logic) → *Explain*, *Wiring diagram*, live simulation.

## Design rule: AI never does the maths
```
image ─► OpenCV (wires, symbols)  ─► OCR (labels, values) ─► structured model ─► deterministic engine
                                                                  │                 (MNA solver / ladder scan)
                                                                  └──► AI (optional): identify unknown symbols,
                                                                       explain the system, answer questions
```
* Circuits: modified nodal analysis; diodes/LEDs use a constant-Vf model solved by iteration. Undefined circuits
  (e.g. a current source feeding a node with no path to ground) are **rejected with a reason**, never solved to nonsense.
* Q&A: "what if R2 were 10 Ω?" → the model only outputs `{"R2": 10}`; the solver computes; the model explains the solver's numbers.
* Ladder: the parser's structure is passed to the model as ground truth when it is confident (no warnings);
  otherwise the model reads the image itself and says how sure it is.

## Run
Windows: double-click `run.bat` (or `py -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`, `python -m uvicorn circuitsense.api:app`).
```bash
./run.sh                      # venv + deps + http://127.0.0.1:8000
export GEMINI_API_KEY=...     # or put GEMINI_API_KEY=... in a local .env file (git-ignored; copy .env.example). Optional: enables Ask, Explain, unknown-symbol fallback (GEMINI_MODEL to force a model)
```
Tests: `pip install -r requirements-dev.txt && pytest` (58 tests, ~6 s; synthetic drawings are generated in `tests/gen.py`).

## Deploy
```bash
docker build -t circuitsense . && docker run -p 8000:8000 -e GEMINI_API_KEY=... circuitsense      # or any host that runs a Dockerfile (Render, Railway, Fly.io, Hugging Face Spaces)
```
The server reads `PORT` and binds `0.0.0.0` in the container. Give the host ~1 GB RAM (OpenCV + the OCR model). `GET /health` is the health check.
Do not commit `.env` or a key. On a public demo set `GEMINI_API_KEY` as a host secret; the AI endpoints are rate-limited per IP
(`RATE_LIMIT_AI`, default 30/hour) so strangers can't drain the quota. Without a key everything except Ask / Explain / the unknown-symbol fallback still works.

## API
`POST /analyze?kind=circuit|ladder` (image) · `POST /solve` · `POST /ac` · `POST /ac/sweep` · `POST /import` · `GET /samples` · `GET /samples/{name}` · `POST /whatif` · `POST /find` · `POST /ask` · `POST /ladder/sim` ·
`POST /ladder/explain` (image + `ladder` form field) · `POST /ladder/wiring` · `POST /wiring` · `GET /health`. Interactive docs at `/docs`.

## Known limits (honest list)
* Wires must be horizontal/vertical. Clean screenshots and textbook figures work; hand-drawn and photographed diagrams do not yet.
* Symbol set is fixed (see table); anything else is flagged `UNKNOWN` and can be identified by the AI fallback or fixed in the UI.
* OCR sometimes reads Ω as a digit (10 Ω → 100); values are editable before solving. Bare wire crossings have a toggle
  ("crossing wires connect") because textbooks disagree.
* Low-resolution ladder images may not parse; the AI explanation still works from the image.
* Ladder timers/counters/function blocks are not parsed. Transistors/op-amps are **not detected from images** (they arrive as UNKNOWN: set the kind and params in the table) but import exactly from netlists. Transistor models are simple piece-wise/level-1 (no Early effect for BJTs, no capacitances); op-amps are ideal unless rails are given; AC covers linear R/C/L + ideal op-amps only (no small-signal transistor models yet).
* Ladder → wiring targets a 24 V DC PLC with sourcing outputs; device tags (S/B/H/KM/K) are guessed from the signal names, and the motor's power circuit is not drawn. Hard-wired relay panels are a different diagram.
* Wiring: connectivity analysis on the schematic parser; box-and-terminal wiring drawings need their own parser.
