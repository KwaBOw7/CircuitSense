"""Q&A about a solved circuit. The AI never calculates: hypotheticals become value changes that the deterministic solver runs."""
import json
from . import ai_fallback
from .model import Circuit
from .solver import solve, whatif
from . import inverse

PLAN = """You explain a solved DC circuit. The data below is ground truth from a deterministic solver.
Never do arithmetic yourself and never invent numbers. Units are SI (ohm, volt, amp, watt). Node 'ground' is 0 V.
Terminals: Q [collector, base, emitter], M [drain, gate, source], OA [out, +in, -in], E [out, ref, ctrl+, ctrl-]. Transistor/op-amp results carry a 'region' (cutoff/active/saturation/triode/linear/saturated).\nComponent nodes: V/I sources list [first, second]; V: first node is +; I: current flows first -> second through the source.
Return JSON only: {"answer": string, "changes": object, "constraints": array}.
If the user asks a hypothetical (e.g. "what if R2 is 10 ohms"), set "changes" to {"R2": 10} (SI values) and "answer" to "".
If the user gives measurements and asks for unknown values (e.g. "R2 has 3 V across it, what is R1?"), set "constraints" to
[{"kind": "v"|"i"|"p"|"node", "of": part id or node number, "value": number}] (v = voltage across the part, i = current through it, p = power) and "answer" to "".
Otherwise "changes" is {}, "constraints" is [] and "answer" explains using only the data. If the data can't answer it, say so."""

EXPLAIN = """A what-if was run by a deterministic solver. In 2-4 sentences say what changed and why, using ONLY the numbers below.
Never compute anything yourself. Plain text."""

def ask(circuit: dict, question: str, gen=None):
    gen = gen or ai_fallback.generate
    c = Circuit.from_dict(circuit)
    try: base = solve(c)
    except ValueError as e: base = {"unsolved": str(e)}         # e.g. unknown values: the model can still plan a /find
    ctx = json.dumps({"components": [{"id": x.id, "kind": x.kind, "nodes": x.nodes, "value": x.value, **({"params": x.params} if x.params else {})} for x in c.components],
                      "results": base}, default=float)
    plan = gen(f"{PLAN}\n\nDATA:\n{ctx}\n\nQUESTION: {question}", as_json=True)
    cons = [k for k in (plan.get("constraints") or []) if isinstance(k, dict)]
    if cons:
        f = inverse.find(c, cons)
        text = gen(f"{EXPLAIN}\n\nQUESTION: {question}\nRESULT (unknown values found by the solver from the given measurements):\n{json.dumps(f, default=float)}")
        return {"answer": text, "found": f, "whatif": None}
    ch = {k: v for k, v in (plan.get("changes") or {}).items() if isinstance(v, (int, float))}
    if not ch: return {"answer": plan.get("answer", ""), "whatif": None}   # (base may be {"unsolved": ...} - whatif then raises a clear error)
    w = whatif(c, ch)
    text = gen(f"{EXPLAIN}\n\nQUESTION: {question}\nRESULT:\n{json.dumps(w, default=float)}")
    return {"answer": text, "whatif": w}
