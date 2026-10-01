"""Connectivity analysis on a parsed netlist: nets, continuity, and wiring faults. Deterministic."""
NAMES = {"V": ["+", "-"], "D": ["A", "K"], "LED": ["A", "K"], "I": ["in", "out"], "Q": ["C", "B", "E"], "M": ["D", "G", "S"], "OA": ["out", "+", "-"], "E": ["out", "ref", "c+", "c-"]}

def analyze(c, a=None, b=None):
    nets = {}
    for x in c.components:
        nm = NAMES.get(x.kind)
        for i, n in enumerate(x.nodes): nets.setdefault(n, []).append(f"{x.id}.{nm[i] if nm else i + 1}")
    where = {t: n for n, ts in nets.items() for t in ts}
    issues = []
    for n, ts in nets.items():
        if len(ts) == 1: issues.append(f"Dangling: {ts[0]} connects to nothing else (net {n})")
    for x in c.components:
        if x.kind != "GND" and len(x.nodes) == 2 and x.nodes[0] == x.nodes[1]:
            issues.append(f"{x.id} has both terminals on the same net" + (" - SOURCE SHORT CIRCUIT" if x.kind in ("V",) else " (bypassed)"))
    r = {"nets": {str(n): ts for n, ts in nets.items()}, "issues": issues}
    if a and b:
        if a not in where or b not in where: r["continuity"] = {"error": f"unknown terminal: {a if a not in where else b}. Terminals: {sorted(where)}"}
        else: r["continuity"] = {"a": a, "b": b, "connected": where[a] == where[b], "net": where[a] if where[a] == where[b] else None}
    return r
