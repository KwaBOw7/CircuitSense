"""Deterministic AC steady-state analysis (phasors, complex MNA) + frequency sweep. R, C, L, V, I only.
Amplitudes are PEAK values; power is average real power P = 1/2 Re(V I*) and reactive Q = 1/2 Im(V I*)."""
import cmath, math
import numpy as np

def _q(v): return float(f"{v:.8g}")
def _ph(z): return {"mag": _q(abs(z)), "phase": _q(math.degrees(cmath.phase(z)))}

def ac_solve(c, f):
    comps = c.components
    bad = [x.id for x in comps if x.kind == "UNKNOWN"]
    if bad: raise ValueError(f"Unresolved components: {bad}")
    nl = [x.id for x in comps if x.kind in ("D", "LED")]
    if nl: raise ValueError(f"Diodes/LEDs are non-linear and not supported in AC analysis yet: {nl}")
    tr = [x.id for x in comps if x.kind in ("Q", "M")]
    if tr: raise ValueError(f"Transistors need small-signal (hybrid-pi) models, which are not supported in AC analysis yet: {tr}")
    if not (f and f > 0): raise ValueError("Frequency must be above 0 Hz (use the DC solver for DC)")
    need = [x.id for x in comps if x.kind in ("R", "C", "L", "V", "I", "E") and (x.value is None or (x.kind in ("R", "C", "L") and x.value <= 0))]
    if need: raise ValueError(f"Missing or non-positive values for: {need}")
    gnd = next((x.nodes[0] for x in comps if x.kind == "GND"), None)
    if gnd is None:
        v0 = next((x for x in comps if x.kind == "V"), None)
        if v0 is None: raise ValueError("No ground reference: add a ground symbol or a voltage source")
        gnd = v0.nodes[1]
    nodes = sorted({n for x in comps for n in x.nodes} - {gnd}); ix = {n: i for i, n in enumerate(nodes)}
    seen, todo = {gnd}, [gnd]                                       # capacitors DO conduct at AC
    while todo:
        n = todo.pop()
        for x in comps:
            if x.kind in ("R", "C", "L", "V", "E") and n in x.nodes[:2]:
                for m in x.nodes[:2]:
                    if m not in seen: seen.add(m); todo.append(m)
    for x in comps:                                                 # an op-amp output is driven, so it is reachable
        if x.kind == "OA" and x.nodes[0] != gnd: seen.add(x.nodes[0])
    stuck = sorted({n for x in comps if x.kind == "I" for n in x.nodes[:2]} - seen)
    if stuck: raise ValueError(f"Current source feeds node(s) {stuck} with no path to ground - check the wiring")
    w = 2 * math.pi * f; src = [x for x in comps if x.kind in ("V", "E", "OA")]; N, M = len(nodes), len(src)
    A, z = np.zeros((N + M, N + M), complex), np.zeros(N + M, complex)
    Y = lambda x: {"R": 1 / x.value, "C": 1j * w * x.value, "L": 1 / (1j * w * x.value)}[x.kind]
    ph = lambda x: cmath.exp(1j * math.radians(x.phase or 0.0))
    for x in comps:
        if x.kind in ("R", "C", "L"):
            y = Y(x); a, b = (ix.get(n) for n in x.nodes[:2])
            for p, q, s in ((a, a, y), (b, b, y), (a, b, -y), (b, a, -y)):
                if p is not None and q is not None: A[p, q] += s
        elif x.kind == "I":
            a, b = (ix.get(n) for n in x.nodes[:2]); cur = x.value * ph(x)
            if a is not None: z[a] -= cur
            if b is not None: z[b] += cur
    for k, x in enumerate(src):
        r = N + k
        if x.kind == "V":
            a, b = (ix.get(n) for n in x.nodes)
            if a is not None: A[a, r] += 1; A[r, a] += 1
            if b is not None: A[b, r] -= 1; A[r, b] -= 1
            z[r] = x.value * ph(x)
        else:                                                       # E: out-ref = gain*(cp-cn)   |   OA: ideal nullor, in+ = in-
            o, rf, cp, cn = (ix.get(n) for n in (x.nodes + [gnd, gnd, gnd])[:4]) if x.kind == "E" else (ix.get(x.nodes[0]), None, ix.get(x.nodes[1]), ix.get(x.nodes[2]))
            g = x.value if x.kind == "E" else None
            if o is not None: A[o, r] += 1
            if x.kind == "E":
                if rf is not None: A[rf, r] -= 1
                if o is not None: A[r, o] += 1
                if rf is not None: A[r, rf] -= 1
                if cp is not None: A[r, cp] -= g
                if cn is not None: A[r, cn] += g
            else:
                if cp is not None: A[r, cp] += 1
                if cn is not None: A[r, cn] -= 1
    try: s = np.linalg.solve(A, z)
    except np.linalg.LinAlgError: raise ValueError("The circuit is singular at this frequency (e.g. a loop of ideal sources, or an ideal L-C resonance)")
    v = lambda n: 0j if n == gnd else complex(s[ix[n]])
    out = {"freq": f, "ground": gnd, "node_v": {n: _ph(v(n)) for n in [gnd] + nodes}, "parts": {}}
    for x in comps:
        if x.kind == "GND": continue
        d = v(x.nodes[0]) - v(x.nodes[1])
        i = d * Y(x) if x.kind in ("R", "C", "L") else (-complex(s[N + src.index(x)]) if x.kind in ("V", "E") else (x.value * ph(x) if x.kind == "I" else -complex(s[N + src.index(x)])))
        if x.kind == "OA": d = v(x.nodes[1]) - v(x.nodes[2])
        sp = 0.5 * d * i.conjugate()
        out["parts"][x.id] = {"v": _ph(d), "i": _ph(i), "p": _q(sp.real), "q": _q(sp.imag)}
    return out

def sweep(c, of, f0, f1, n=120):
    """of = 'node:3' | 'v:R1' | 'i:C1'. Returns magnitude/phase vs frequency and the cutoff or resonance if there is one."""
    kind, _, target = of.partition(":")
    if kind not in ("node", "v", "i") or not target: raise ValueError("'of' must look like node:3, v:R1 or i:C1")
    if not (0 < f0 < f1) or not (2 <= n <= 2000): raise ValueError("Need 0 < f0 < f1 and 2-2000 points")
    fs = np.logspace(math.log10(f0), math.log10(f1), int(n)); mag, phase = [], []
    for f in fs:
        r = ac_solve(c, float(f))
        try: m = r["node_v"][int(target)] if kind == "node" else r["parts"][target][kind]
        except (KeyError, ValueError): raise ValueError(f"Unknown target: {of}")
        mag.append(m["mag"]); phase.append(m["phase"])
    m = np.array(mag); src = next((x for x in c.components if x.kind == "V" and x.value), None)
    ref = src.value if (src and kind in ("node", "v")) else None                # gain relative to the source amplitude
    db = [_q(20 * math.log10(max(x / (ref or 1), 1e-30))) for x in m]
    res = {"of": of, "f": [_q(x) for x in fs], "mag": mag, "phase": phase, "db": db, "db_ref": ref, "f_peak": None, "f_corner": None, "bandwidth": None}
    k = int(np.argmax(m)); ends = max(m[0], m[-1])
    lg = lambda i, j, t: float(10 ** (math.log10(fs[i]) + (math.log10(fs[j]) - math.log10(fs[i])) * t))
    def cross(level, rng):
        for i in rng:
            if (m[i] - level) * (m[i + 1] - level) < 0: return lg(i, i + 1, (level - m[i]) / (m[i + 1] - m[i]))
    if 0 < k < len(m) - 1 and m[k] > 1.05 * ends:                                # resonance: peak in the interior
        res["f_peak"] = _q(fs[k]); lo, hi = cross(m[k] / math.sqrt(2), range(k - 1, -1, -1)), cross(m[k] / math.sqrt(2), range(k, len(m) - 1))
        if lo and hi: res["bandwidth"] = _q(hi - lo)
    elif ends > 0:
        c_ = cross(ends / math.sqrt(2), range(len(m) - 1)); res["f_corner"] = _q(c_) if c_ else None
    return res
