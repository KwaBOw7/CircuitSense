"""Find unknown component values from known measurements (deterministic numerical inverse of the DC solver).
Unknowns = R/V/I parts with value None. Constraints: {"kind": "v"|"i"|"p"|"node", "of": part id | node number, "value": number}
  v = voltage across a part (nodes[0] - nodes[1]), i = current through it (nodes[0] -> nodes[1]; out of + for sources), p = power, node = node voltage."""
import copy
import numpy as np
from .solver import solve

def _get(res, k):
    return res["node_v"][int(k["of"])] if k["kind"] == "node" else res["parts"][k["of"]][k["kind"]]

def find(c, constraints):
    unk = [x for x in c.components if x.kind in ("R", "V", "I") and x.value is None]
    if not unk: raise ValueError("Every value is already known - nothing to find")
    if not constraints: raise ValueError("Give at least one measurement (e.g. the voltage across a resistor)")
    ids = {x.id for x in c.components}; nodes = {n for x in c.components for n in x.nodes}
    for k in constraints:
        if k.get("kind") not in ("v", "i", "p", "node") or not isinstance(k.get("value"), (int, float)): raise ValueError(f"Bad measurement: {k}")
        if (k["kind"] == "node" and int(k["of"]) not in nodes) or (k["kind"] != "node" and k["of"] not in ids): raise ValueError(f"Unknown target in {k}")
    tgt = np.array([k["value"] for k in constraints], float); scale = np.maximum(np.abs(tgt), 1e-9)

    def fill(th):
        c2 = copy.deepcopy(c); m = {x.id: x for x in c2.components}
        for u, t in zip(unk, th): m[u.id].value = float(np.exp(t)) if u.kind == "R" else float(t)   # resistances stay positive
        return c2
    def resid(th):
        try:
            r = solve(fill(th)); return (np.array([_get(r, k) for k in constraints]) - tgt) / scale
        except (ValueError, np.linalg.LinAlgError, KeyError, FloatingPointError):
            return np.full(len(constraints), 1e3)
    def lm(x):
        r = resid(x); lam = 1e-2
        for _ in range(120):
            J = np.empty((len(r), len(x)))
            for j in range(len(x)):
                h = 1e-6 * max(1, abs(x[j])); xp = x.copy(); xp[j] += h; J[:, j] = (resid(xp) - r) / h
            A, g = J.T @ J, J.T @ r
            while True:
                dx = -np.linalg.solve(A + lam * np.eye(len(x)), g); rn = resid(x + dx)
                if rn @ rn < r @ r: x, r, lam = x + dx, rn, max(lam / 3, 1e-12); break
                lam *= 4
                if lam > 1e12: return x, r, J
            if r @ r < 1e-24: break
        return x, r, J

    rng = np.random.default_rng(0); sols = []
    for _ in range(16):
        x0 = np.array([rng.uniform(np.log(10), np.log(1e7)) if u.kind == "R" else rng.uniform(-10, 10) for u in unk])
        x, r, J = lm(x0)
        if np.max(np.abs(r)) < 1e-6: sols.append((x, J))
    if not sols: raise ValueError("No positive values satisfy these measurements - they are inconsistent with the circuit (or contradict each other)")
    vals = lambda x: [float(np.exp(t)) if u.kind == "R" else float(t) for u, t in zip(unk, x)]
    x, J = sols[0]; sv = np.linalg.svd(J, compute_uv=False)
    rank = int((sv > 1e-7 * sv.max()).sum()) if len(sv) else 0
    distinct = any(not np.allclose(vals(a), vals(x), rtol=1e-3, atol=1e-9) for a, _ in sols[1:])
    unique = rank == len(unk) and not distinct
    out = dict(zip((u.id for u in unk), (float(f"{v:.8g}") for v in vals(x))))
    cn = np.linalg.norm(J, axis=0); blind = [u.id for u, n in zip(unk, cn) if n <= 1e-7 * max(cn.max(), 1e-12) or cn.max() == 0]
    msg = "Unique solution." if unique else f"NOT unique: these measurements do not depend on {', '.join(blind)} at all - measure something it affects (a voltage/current in its own branch)." if blind else \
          f"NOT unique: {len(constraints)} measurement(s) cannot pin down {len(unk)} unknown(s) ({', '.join(out)}); one valid answer is shown - add another measurement."
    return {"values": out, "unique": unique, "message": msg, "solution": solve(fill(x))}
