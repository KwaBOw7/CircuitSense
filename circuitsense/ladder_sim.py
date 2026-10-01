"""Deterministic ladder scan + boolean equations."""
def _edges(lad, state):
    for e in lad["elements"]:
        a = state.get(e["addr"], False)
        if (e["type"] == "NO" and a) or (e["type"] == "NC" and not a): yield e["nodes"]

def _reach(lad, state, start=1):
    seen, todo, ed = {start}, [start], list(_edges(lad, state))
    while todo:
        n = todo.pop()
        for a, b in ed:
            for u, v in ((a, b), (b, a)):
                if u == n and v not in seen: seen.add(v); todo.append(v)
    return seen

def scan(lad, state, inputs):
    state = {**state, **inputs}
    coils = sorted((e for e in lad["elements"] if e["type"] == "COIL"), key=lambda e: (e["y"], e["x"]))
    for _ in range(20):                                          # scan until stable (seal-ins settle)
        prev = dict(state)
        for c in coils:
            en = c["nodes"][0] in _reach(lad, state)
            if c["coil"] == "latch": state[c["addr"]] = state.get(c["addr"], False) or en
            elif c["coil"] == "unlatch": state[c["addr"]] = state.get(c["addr"], False) and not en
            else: state[c["addr"]] = en
        if state == prev: break
    return state

def io(lad):
    outs = sorted({e["addr"] for e in lad["elements"] if e["type"] == "COIL" and e["addr"]})
    ins = sorted({e["addr"] for e in lad["elements"] if e["type"] in ("NO", "NC") and e["addr"]} - set(outs))
    return ins, outs

def coil_terms(lad, c):
    """Every conducting path rail -> coil as a set of (name, must_be_active) literals."""
    terms = []
    def dfs(n, seen, lits):
        if len(terms) > 200: return
        if n == c["nodes"][0]: terms.append(frozenset(lits)); return
        for e in lad["elements"]:
            if e["type"] not in ("NO", "NC") or n not in e["nodes"]: continue
            m = e["nodes"][1] if e["nodes"][0] == n else e["nodes"][0]
            if m not in seen: dfs(m, seen | {m}, lits | {(e["addr"], e["type"] == "NO")})
    dfs(1, {1}, frozenset())
    terms = [t for t in set(terms) if not any((a, not p) in t for a, p in t)]     # drop x·¬x
    return [t for t in terms if not any(o < t for o in terms)]                    # absorption

def equations(lad):
    out = []
    for c in (e for e in lad["elements"] if e["type"] == "COIL" and e["addr"]):
        fmt = lambda t: "·".join(sorted(a if p else f"¬{a}" for a, p in t)) or "1"
        rhs = " + ".join(sorted(fmt(t) for t in coil_terms(lad, c))) or "0"
        out.append(f"{c['addr']} {'SET when' if c['coil'] == 'latch' else 'RESET when' if c['coil'] == 'unlatch' else '='} {rhs}")
    return out

def rung_text(lad):
    """Plain-English description of each output rung (deterministic)."""
    out, nums = [], lad.get("rung_numbers") or []
    def rung_no(c, n):                                            # use the numbers printed beside the rails when readable
        b = min(nums, key=lambda q: abs(q[1] - c["y"]), default=None)
        return b[0] if b and abs(b[1] - c["y"]) <= .06 * lad.get("height", 1e9) else n
    for n, c in enumerate(sorted((e for e in lad["elements"] if e["type"] == "COIL"), key=lambda e: (e["y"], e["x"])), 1):
        a = c["addr"] or "?"; n = rung_no(c, n)
        act = {"latch": f"{a} is LATCHED ON (stays on until an unlatch coil resets it)", "unlatch": f"{a} is UNLATCHED (reset to off)"}.get(c["coil"], f"{a} is energised")
        ts = [" AND ".join(sorted(f"{x} is active" if p else f"{x} is NOT active" for x, p in t)) or "always" for t in coil_terms(lad, c)]
        out.append(f"Rung {n}: {act} when " + " OR ".join(f"({t})" if len(ts) > 1 else t for t in sorted(ts)) if ts else f"Rung {n}: {act} - but no conducting path from the rail was found")
    return out
