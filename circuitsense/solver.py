"""Deterministic DC operating-point solver (modified nodal analysis).
Linear: R, V, I, L (short), C (open).  Piece-wise linear, solved by iterating device states: diode/LED (constant Vf),
BJT (Vbe=0.7 V, beta, Vce(sat)=0.2 V; cutoff/active/saturation), ideal op-amp (with optional supply rails), VCVS.
Non-linear: MOSFET level-1 square law (Newton).   Terminals: Q [C,B,E]  M [D,G,S]  OA [out,+,-]  E [out,ref,cp,cn]."""
import copy
import numpy as np

VON, VCESAT, BETA = 0.7, 0.2, 100.0
LIN2 = ("R", "V", "L", "D", "LED")

def _p(x, k, d):
    v = (x.params or {}).get(k); return d if v is None else v

def _pol(x):     # +1 NPN / N-channel, -1 PNP / P-channel
    return -1 if str((x.params or {}).get("polarity", (x.params or {}).get("type", "N")))[:1].upper() in ("P",) else 1

def _mos(vgs, vds, K, vth, lam):
    """N-channel level-1: returns Id, gm, gds for vds >= 0."""
    vov = vgs - vth
    if vov <= 0: return "cutoff", 0.0, 0.0, 0.0
    if vds < vov: return "triode", K * (vov * vds - vds * vds / 2), K * vds, K * (vov - vds)
    return "saturation", 0.5 * K * vov * vov * (1 + lam * vds), K * vov * (1 + lam * vds), 0.5 * K * vov * vov * lam

def _reach(comps, gnd):
    seen, todo, edges = {gnd}, [gnd], []
    for x in comps:
        n, k = x.nodes, x.kind
        if k in LIN2 or k == "E": edges.append((n[0], n[1]))
        elif k == "OA": edges.append((n[0], gnd))
        elif k == "Q": edges += [(n[0], n[2]), (n[1], n[2])]
        elif k == "M": edges.append((n[0], n[2]))
    while todo:
        a = todo.pop()
        for u, v in edges:
            for p, q in ((u, v), (v, u)):
                if p == a and q not in seen: seen.add(q); todo.append(q)
    return seen

def solve(c):
    comps = c.components
    bad = [x.id for x in comps if x.kind == "UNKNOWN"]
    if bad: raise ValueError(f"Unresolved components: {bad}")
    need = [x.id for x in comps if x.kind in ("R", "V", "I", "E") and x.value is None]
    if need: raise ValueError(f"Missing values for: {need}")
    gnd = next((x.nodes[0] for x in comps if x.kind == "GND"), None)
    if gnd is None:
        v = next((x for x in comps if x.kind == "V"), None)
        if v is None: raise ValueError("No ground reference: add a ground symbol or a voltage source")
        gnd = v.nodes[1]
    nodes = sorted({n for x in comps for n in x.nodes} - {gnd}); ix = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    seen = _reach(comps, gnd)
    stuck = sorted({n for x in comps if x.kind == "I" for n in x.nodes[:2]} - seen)
    if stuck: raise ValueError(f"Current source feeds node(s) {stuck} that have no path to ground, so the voltage is undefined. "
                               "Usually a wire connection was missed - try ticking 'crossing wires connect' and re-analyze, or fix the nodes.")
    floating = [n for n in nodes if n not in seen]
    vf = lambda x: x.value if x.value is not None else (2.0 if x.kind == "LED" else 0.7)
    S = {"dio": {x.id for x in comps if x.kind in ("D", "LED")}, "q": {x.id: "cut" for x in comps if x.kind == "Q"},   # start open: no false source conflicts
         "oa": {x.id: "lin" for x in comps if x.kind == "OA"}, "mreg": {}}
    has_m = any(x.kind == "M" for x in comps); v = np.zeros(N)
    V = lambda vec, n: 0.0 if n == gnd else float(vec[ix[n]])

    for it in range(120):
        br = []                                                                  # (tag, comp) voltage-type branches
        for x in comps:
            if x.kind in ("V", "L") or (x.kind in ("D", "LED") and x.id in S["dio"]): br.append(("V", x))
            elif x.kind == "E": br.append(("E", x))
            elif x.kind == "OA": br.append(("OA", x))
            elif x.kind == "Q":
                if S["q"][x.id] != "cut": br.append(("QB", x))
                if S["q"][x.id] == "sat": br.append(("QC", x))
        M = len(br); A, z = np.zeros((N + M, N + M)), np.zeros(N + M); I_ = lambda n: ix.get(n)
        def add(r, cidx, val):
            if r is not None and cidx is not None: A[r, cidx] += val
        for x in comps:
            n = x.nodes
            if x.kind == "R":
                g = 1 / x.value; a, b = I_(n[0]), I_(n[1])
                for p_, q_, s_ in ((a, a, g), (b, b, g), (a, b, -g), (b, a, -g)): add(p_, q_, s_)
            elif x.kind == "I":
                a, b = I_(n[0]), I_(n[1])
                if a is not None: z[a] -= x.value
                if b is not None: z[b] += x.value
            elif x.kind == "M":                                               # Newton companion model around the previous iterate
                pol = _pol(x); d, g_, s_ = n; K = _p(x, "kp", 2e-3); vth = abs(_p(x, "vth", 1.0)); lam = _p(x, "lam", 0.0)
                if pol * (V(v, d) - V(v, s_)) < 0: d, s_ = s_, d                # source/drain swap
                vgs, vds = pol * (V(v, g_) - V(v, s_)), pol * (V(v, d) - V(v, s_))
                reg, idr, gm, gds = _mos(vgs, vds, K, vth, lam); S["mreg"][x.id] = reg
                I0 = pol * idr; Ieq = I0 - gm * V(v, g_) - gds * V(v, d) + (gm + gds) * V(v, s_)
                D_, G_, S2 = I_(d), I_(g_), I_(s_)
                for row, sg in ((D_, 1), (S2, -1)):
                    add(row, G_, sg * gm); add(row, D_, sg * gds); add(row, S2, -sg * (gm + gds))
                    if row is not None: z[row] -= sg * Ieq
        idx = {}
        for k, (tag, x) in enumerate(br):
            r = N + k; idx[(tag, x.id)] = r; n = x.nodes
            if tag == "V":
                a, b = I_(n[0]), I_(n[1]); add(a, r, 1); add(r, a, 1); add(b, r, -1); add(r, b, -1)
                z[r] = x.value if x.kind == "V" else (0.0 if x.kind == "L" else vf(x))
            elif tag == "E":
                o, rf, cp, cn = (I_(t) for t in n[:4]); add(o, r, 1); add(rf, r, -1); add(r, o, 1); add(r, rf, -1); add(r, cp, -x.value); add(r, cn, x.value)
            elif tag == "OA":
                o, ip, im = (I_(t) for t in n[:3]); add(o, r, 1); st = S["oa"][x.id]
                if st == "lin": add(r, ip, 1); add(r, im, -1)
                else: add(r, o, 1); z[r] = _p(x, "vpos" if st == "hi" else "vneg", 0.0)
            elif tag in ("QB", "QC"):
                C_, B_, E_ = n; pol = _pol(x)
                if tag == "QB":
                    a, b = (I_(B_), I_(E_)) if pol > 0 else (I_(E_), I_(B_)); add(a, r, 1); add(r, a, 1); add(b, r, -1); add(r, b, -1); z[r] = VON
                    if S["q"][x.id] == "act":                                     # collector current = beta * base current
                        ca, cb = (I_(C_), I_(E_)) if pol > 0 else (I_(E_), I_(C_)); beta = _p(x, "beta", BETA); add(ca, r, beta); add(cb, r, -beta)
                else:
                    a, b = (I_(C_), I_(E_)) if pol > 0 else (I_(E_), I_(C_)); add(a, r, 1); add(r, a, 1); add(b, r, -1); add(r, b, -1); z[r] = VCESAT
        for nn in floating: A[ix[nn], ix[nn]] += 1e-12                           # only nodes with no DC path get a tiny leak
        try: sol = np.linalg.solve(A, z)
        except np.linalg.LinAlgError: raise ValueError("The circuit is singular (e.g. a loop of ideal sources, or an op-amp/source conflict)")
        vn = sol[:N]; ch = False
        for x in comps:
            n = x.nodes
            if x.kind in ("D", "LED"):
                if x.id in S["dio"] and sol[idx[("V", x.id)]] < -1e-12: S["dio"].discard(x.id); ch = True
                elif x.id not in S["dio"] and V(vn, n[0]) - V(vn, n[1]) > vf(x) + 1e-9: S["dio"].add(x.id); ch = True
            elif x.kind == "Q":
                pol = _pol(x); st = S["q"][x.id]; ib = sol[idx[("QB", x.id)]] if ("QB", x.id) in idx else 0.0
                vbe, vce = pol * (V(vn, n[1]) - V(vn, n[2])), pol * (V(vn, n[0]) - V(vn, n[2])); new = st
                if st == "cut" and vbe > VON + 1e-9: new = "act"
                elif st == "act": new = "cut" if ib < -1e-12 else ("sat" if vce < VCESAT - 1e-9 else "act")
                elif st == "sat":
                    ic = sol[idx[("QC", x.id)]]; new = "cut" if ib < -1e-12 else ("act" if ic > _p(x, "beta", BETA) * ib + 1e-12 else "sat")
                if new != st: S["q"][x.id] = new; ch = True
            elif x.kind == "OA":
                st = S["oa"][x.id]; vo = V(vn, n[0]); diff = V(vn, n[1]) - V(vn, n[2]); vp, vm = x.params.get("vpos"), x.params.get("vneg"); new = st
                if st == "lin": new = "hi" if (vp is not None and vo > vp + 1e-9) else "lo" if (vm is not None and vo < vm - 1e-9) else "lin"
                elif st == "hi" and diff <= 0: new = "lin"
                elif st == "lo" and diff >= 0: new = "lin"
                if new != st: S["oa"][x.id] = new; ch = True
        dv = float(np.max(np.abs(vn - v))) if N else 0.0
        if has_m and it > 25: vn = 0.5 * (vn + v)                                  # damping if Newton is wandering
        v = vn
        if not ch and (not has_m or dv < 1e-9): break
    else:
        raise ValueError("The non-linear circuit did not settle to one operating point (check transistor bias and polarity)")

    q = lambda t: float(f"{t:.10g}"); vv = lambda n: V(v, n)
    out = {"ground": gnd, "node_v": {n: q(vv(n)) for n in [gnd] + nodes}, "parts": {}}
    for x in comps:
        n = x.nodes
        if x.kind in ("GND", "UNKNOWN"): continue
        if x.kind == "Q":
            pol = _pol(x); st = S["q"][x.id]; ib = float(sol[idx[("QB", x.id)]]) if st != "cut" else 0.0
            ic = float(sol[idx[("QC", x.id)]]) if st == "sat" else (_p(x, "beta", BETA) * ib if st == "act" else 0.0)
            vbe, vce = vv(n[1]) - vv(n[2]), vv(n[0]) - vv(n[2])
            out["parts"][x.id] = {"v": q(vce), "i": q(pol * ic), "p": q(vce * pol * ic + vbe * pol * ib), "vbe": q(vbe), "ib": q(pol * ib),
                                  "region": {"cut": "cutoff", "act": "active", "sat": "saturation"}[st]}
            continue
        if x.kind == "M":
            pol = _pol(x); d, s_ = (n[0], n[2]); vds_signed = vv(d) - vv(s_)
            reg, idr, _, _ = _mos(pol * (vv(n[1]) - vv(s_)) if pol * vds_signed >= 0 else pol * (vv(n[1]) - vv(d)), abs(vds_signed), _p(x, "kp", 2e-3), abs(_p(x, "vth", 1.0)), _p(x, "lam", 0.0))
            idr = pol * idr * (1 if pol * vds_signed >= 0 else -1)
            out["parts"][x.id] = {"v": q(vds_signed), "i": q(idr), "p": q(vds_signed * idr), "vgs": q(vv(n[1]) - vv(s_)), "region": reg}
            continue
        if x.kind == "OA":
            io = -float(sol[idx[("OA", x.id)]]); out["parts"][x.id] = {"v": q(vv(n[1]) - vv(n[2])), "i": q(io), "p": q(io * vv(n[0])), "vout": q(vv(n[0])),
                                                                       "region": {"lin": "linear", "hi": "saturated high", "lo": "saturated low"}[S["oa"][x.id]]}
            continue
        d = vv(n[0]) - vv(n[1])
        if x.kind == "R": i = d / x.value
        elif x.kind in ("V", "L"): i = float(-sol[idx[("V", x.id)]])
        elif x.kind == "E": i = float(-sol[idx[("E", x.id)]])
        elif x.kind in ("D", "LED"): i = float(sol[idx[("V", x.id)]]) if x.id in S["dio"] else 0.0
        elif x.kind == "I": i = x.value
        else: i = 0.0
        out["parts"][x.id] = {"v": q(d), "i": q(i), "p": q(d * i), **({"on": x.id in S["dio"]} if x.kind in ("D", "LED") else {})}
    return out

def whatif(c, changes):
    """Re-solve with changed component values (SI units). Baseline circuit is untouched."""
    ids = {x.id for x in c.components}
    bad = [k for k in changes if k not in ids]
    if bad: raise ValueError(f"Unknown components: {bad}")
    c2 = copy.deepcopy(c)
    for x in c2.components:
        if x.id in changes: x.value = float(changes[x.id])
    return {"changes": changes, "before": solve(c), "after": solve(c2)}
