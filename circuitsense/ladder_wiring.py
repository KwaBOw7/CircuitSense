"""Ladder program -> PLC I/O wiring diagram (deterministic).
Inputs  = contact addresses no coil drives  -> field devices wired  24V+ -> device -> DIn
Outputs = coil addresses                     -> loads wired          DOn -> load -> 0V
Conventions (IEC letters): S switch, B sensor, H lamp, KM motor contactor, K relay. Chosen from the NAME; change them if the guess is wrong."""
import re
from html import escape

def _tag(name, kind):
    n = name.lower()
    if kind == "in": return "B" if re.search(r"sensor|level|prox|photo|limit|switch\s*\(?sensor", n) else "S"
    if re.search(r"motor|pump|fan|compressor", n): return "KM"
    if re.search(r"lamp|light|led|alarm|horn|buzzer|^[grybw]$|green|red|yellow|amber", n): return "H"
    return "K"

def build(lad, internal=()):
    els = lad["elements"]; internal = {i.lower() for i in internal}
    coils = {e["addr"] for e in els if e["type"] == "COIL" and e["addr"]}
    order = lambda e: (e["y"], e["x"])
    uses, seen = {}, []
    for e in sorted(els, key=order):
        if e["type"] in ("NO", "NC") and e["addr"] and e["addr"] not in coils:
            uses.setdefault(e["addr"], set()).add(e["type"])
            if e["addr"] not in seen: seen.append(e["addr"])
    outs, used_as_contact = [], {e["addr"] for e in els if e["type"] in ("NO", "NC")}
    for e in sorted(els, key=order):
        if e["type"] == "COIL" and e["addr"] and e["addr"] not in outs and e["addr"].lower() not in internal: outs.append(e["addr"])
    cnt, wires, ins, loads, notes = {}, [], [], [], []
    def W(a, b): wires.append({"id": f"W{len(wires) + 1}", "from": a, "to": b}); return wires[-1]["id"]
    def tag(name, kind):
        p = _tag(name, kind); cnt[p] = cnt.get(p, 0) + 1; return f"{p}{cnt[p]}"
    for i, name in enumerate(seen):
        t = tag(name, "in"); ct = "NC" if uses[name] == {"NC"} else "NO"
        w1, w2 = W("24V+", f"{t}.1"), W(f"{t}.2", f"DI{i}")
        ins.append({"name": name, "tag": t, "terminal": f"DI{i}", "device_contact": ct, "wires": [w1, w2],
                    "program_uses": sorted(uses[name])})
        if len(uses[name]) > 1: notes.append(f"{name} is used as both NO and NC in the program - one field contact feeds both, which is normal.")
    for i, name in enumerate(outs):
        t = tag(name, "out"); w1, w2 = W(f"DO{i}", f"{t}.A1"), W(f"{t}.A2", "0V")
        loads.append({"name": name, "tag": t, "terminal": f"DO{i}", "wires": [w1, w2], "also_read_by_program": name in used_as_contact})
        if name in used_as_contact: notes.append(f"{name} drives an output AND is read as a contact elsewhere: the program reads the internal bit; add an auxiliary contact only if you need physical feedback.")
        if t == "KM": notes.append(f"{name}: the PLC output drives contactor {t}'s coil. The motor itself is wired through the contactor's power contacts on a separate power circuit (not shown).")
    if any(e["type"] == "UNKNOWN" for e in els): notes.append("The ladder contains UNKNOWN elements - the wiring ignores them. Fix them first.")
    notes.append("Assumes a 24 V DC PLC with sourcing outputs. Field-device contact type follows how the program uses the input; check it against the real device (stop/limit devices are normally NC).")
    return {"inputs": ins, "outputs": loads, "wires": wires, "notes": notes}

def svg(w, dark=True):
    ink, mut, acc = ("#cfd8df", "#7f8d98", "#e6a23c") if dark else ("#111", "#555", "#a05a00")
    ni, no = len(w["inputs"]), len(w["outputs"]); rows = max(ni, no, 1); H = 130 + rows * 76; Wd = 960
    T = lambda x, y, s, c=ink, a="start", sz=12, b="": f'<text x="{x}" y="{y}" fill="{c}" text-anchor="{a}" font-family="monospace" font-size="{sz}" {b}>{escape(str(s))}</text>'
    L = lambda x1, y1, x2, y2, c=ink, d="": f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.6" {d}/>'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {H}" width="100%" style="background:{"transparent" if dark else "#fff"}">']
    top, bot = 60, H - 40; px0, px1 = 380, 580
    o += [L(70, top, 70, bot, acc), T(70, top - 14, "24V+", acc, "middle", 13, 'font-weight="bold"'), L(890, top, 890, bot, mut), T(890, top - 14, "0V", mut, "middle", 13, 'font-weight="bold"')]
    o += [f'<rect x="{px0}" y="{top - 20}" width="{px1 - px0}" height="{bot - top + 40}" fill="none" stroke="{ink}" stroke-width="1.8"/>', T((px0 + px1) / 2, top - 6, "PLC", ink, "middle", 14, 'font-weight="bold"')]
    for i, d in enumerate(w["inputs"]):
        y = top + 40 + i * 76; wa, wb = d["wires"]
        o += [L(70, y, 150, y, acc), f'<circle cx="70" cy="{y}" r="3" fill="{acc}"/>', L(220, y, px0, y), L(px0, y, px0 + 8, y),
              T(185, y - 30, f'{d["tag"]}  {d["name"]}  ({d["device_contact"]})', ink, "middle", 11), T(110, y - 5, wa, acc, "middle", 10), T(300, y - 5, wb, acc, "middle", 10), T(px0 + 12, y + 4, d["terminal"], ink, "start", 11)]
        o += [f'<circle cx="150" cy="{y}" r="3.5" fill="none" stroke="{ink}"/><circle cx="220" cy="{y}" r="3.5" fill="none" stroke="{ink}"/>']
        o += [L(150, y, 214, y - 22) if d["device_contact"] == "NO" else L(150, y, 220, y - 14), ] + ([L(196, y - 22, 196, y - 3)] if d["device_contact"] == "NC" else [])
    for i, d in enumerate(w["outputs"]):
        y = top + 40 + i * 76; wa, wb = d["wires"]; sq = d["tag"].startswith("K")   # K / KM = coil symbol, H = lamp
        o += [L(px1 - 8, y, 690, y), L(770, y, 890, y, mut), f'<circle cx="890" cy="{y}" r="3" fill="{mut}"/>', T(px1 - 12, y + 4, d["terminal"], ink, "end", 11),
              T(640, y - 5, wa, acc, "middle", 10), T(830, y - 5, wb, acc, "middle", 10), T(730, y - 30, f'{d["tag"]}  {d["name"]}', ink, "middle", 11)]
        o += [f'<rect x="690" y="{y - 16}" width="80" height="32" fill="none" stroke="{ink}" stroke-width="1.6"/>' if sq else f'<circle cx="730" cy="{y}" r="16" fill="none" stroke="{ink}" stroke-width="1.6"/>',
              T(730, y + 4, "coil" if sq else "lamp", mut, "middle", 10)]
    return "\n".join(o) + "</svg>"

def generate(lad, internal=()):
    w = build(lad, internal); w["svg"] = svg(w, True); w["svg_print"] = svg(w, False); return w
