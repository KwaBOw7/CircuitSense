"""Import a netlist exported from KiCad (SPICE .cir/.sp or the .net S-expression) -> Circuit. Exact connectivity, no image recognition."""
import re
from .model import Circuit, Component

_SUF = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}
_GND = re.compile(r"(^|/)(a|d|p)?gnd[a-z0-9_]*$|^0$", re.I)

def parse_value(t, kind=None):
    """'4k7' -> 4700, '100n' -> 1e-7, '10Meg'/'10M' (resistor) -> 1e7, '5V', '10kΩ', '1e-3'. Returns None if not a number."""
    t = str(t).strip().replace("Ω", "").replace("ohm", "").replace(",", ".")
    m = re.fullmatch(r"(\d+)([fpnuµkKmMgGt])(\d+)", t)                       # KiCad RKM code: 4k7, 2u2
    if m: return float(f"{m.group(1)}.{m.group(3)}") * _SUF[m.group(2).lower()]
    m = re.fullmatch(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(meg|[fpnuµmkgt])?[a-zA-Z]*", t, re.I)
    if not m: return None
    sfx = m.group(2)
    if sfx == "M" and kind == "R": sfx = "meg"                               # KiCad convention: 1M resistor = 1 megohm
    return float(m.group(1)) * (_SUF[sfx.lower()] if sfx else 1)

def _source(text):
    """-> (value, phase_deg, freq, note) from 'DC 10', '5', 'AC 1 90', 'SIN(0 5 1k)', 'dc=10 ampl=5 freq=1k' ..."""
    s = str(text).strip(); note = ""
    m = re.search(r"sin\s*\(\s*([^\s)]+)\s+([^\s)]+)\s*([^\s)]+)?", s, re.I)
    if m: return parse_value(m.group(2)), 0.0, parse_value(m.group(3)) if m.group(3) else None, f"SIN offset {m.group(1)}"
    kv = {k.lower(): v for k, v in re.findall(r"(\w+)\s*=\s*([^\s]+)", s)}
    if kv:
        for k in ("ampl", "amplitude", "ac"):
            if k in kv: return parse_value(kv[k]), parse_value(kv.get("phase", "0")) or 0.0, parse_value(kv.get("freq", "")) if "freq" in kv else None, ""
        if "dc" in kv: return parse_value(kv["dc"]), 0.0, None, ""
    tk = s.replace("(", " ").replace(")", " ").split(); dc = ac = None; phase = 0.0
    for i, w in enumerate(tk):
        lw = w.lower()
        if lw == "dc" and i + 1 < len(tk): dc = parse_value(tk[i + 1])
        elif lw == "ac" and i + 1 < len(tk):
            ac = parse_value(tk[i + 1])
            if i + 2 < len(tk): phase = parse_value(tk[i + 2]) or 0.0
    if ac is not None: return ac, phase, None, (f"DC {dc}" if dc is not None else "")
    if dc is not None: return dc, 0.0, None, ""
    return (parse_value(tk[0]) if tk else None), 0.0, None, note

def _finish(comps, names, warns, freq):
    if any(n == 0 for x in comps for n in x.nodes): comps.append(Component("GND1", "GND", [0]))
    else: warns.append("No ground node found (a net named GND or 0) - the lowest source terminal will be used as 0 V")
    return Circuit(comps, warns, 1.0, {"node_names": {str(k): v for k, v in names.items()}, "freq_hint": freq, "source": "netlist"})

_PNP = {"2N3906", "2N2907", "BC557", "BC327", "BC558", "TIP32", "2N4403", "BD140", "MMBT3906"}
def _guess_pol(model):
    m = model.upper()
    if "PNP" in m or m in _PNP: return "PNP"
    if "NPN" in m or re.match(r"(2N3904|2N2222|BC54|BC33[7-9]|TIP31|BD139|2N4401|MMBT3904|PN2222)", m): return "NPN"
    return None

# ------------------------------------------------ SPICE (.cir / .sp) -------------------------------------------------
def parse_spice(text):
    lines = []
    for raw in text.splitlines():
        s = re.split(r"\s;|\s\$", raw.strip())[0].strip()
        if not s or s.startswith("*"): continue
        if s.startswith("+") and lines: lines[-1] += " " + s[1:].strip()
        else: lines.append(s)
    if lines and not lines[0].startswith(".") and not re.match(r"^[RCLVIDQMXJKEFGHBSWTU]\S*\s+\S+\s+\S+", lines[0], re.I): lines = lines[1:]     # title line
    models = {}
    for s_ in lines:                                                          # .model NAME TYPE(K=V ...)
        m = re.match(r"\.model\s+(\S+)\s+(\w+)\s*\(?(.*?)\)?\s*$", s_, re.I)
        if m: models[m.group(1).lower()] = (m.group(2).upper(), {k.lower(): parse_value(v) for k, v in re.findall(r"(\w+)\s*=\s*([^\s)]+)", m.group(3))})
    ids, names, comps, warns, freq = {}, {}, [], [], None
    def node(nm):
        if _GND.search(nm): ids.setdefault(nm, 0); names[0] = nm; return 0
        if nm not in ids: ids[nm] = len(ids) + (1 if 0 not in ids.values() else 0); names[ids[nm]] = nm
        return ids[nm]
    seq = 0
    for s in lines:
        if s.startswith(".") or s.lower() == "end": continue
        tk = s.split(); name = tk[0]; k0 = name[0].upper()
        if k0 in "RCL" and len(tk) >= 4:
            v = parse_value(tk[3], k0); comps.append(Component(name, k0, [node(tk[1]), node(tk[2])], v))
            if v is None: warns.append(f"{name}: value '{tk[3]}' not understood")
        elif k0 in "VI" and len(tk) >= 3:
            v, ph, fq, note = _source(" ".join(tk[3:])); freq = fq or freq
            comps.append(Component(name, k0, [node(tk[1]), node(tk[2])], v, note=note, phase=ph))
            if v is None: warns.append(f"{name}: source value not understood")
        elif k0 == "D" and len(tk) >= 3:
            comps.append(Component(name, "LED" if "led" in " ".join(tk[3:]).lower() else "D", [node(tk[1]), node(tk[2])], None))
        elif k0 == "Q" and len(tk) >= 5:
            mname = next(t for t in reversed(tk[4:]) if "=" not in t); typ, kv = models.get(mname.lower(), (None, {}))
            pol = typ if typ in ("NPN", "PNP") else _guess_pol(mname)
            if pol is None: pol = "NPN"; warns.append(f"{name}: model '{mname}' has no .model card - assumed NPN, beta 100 (change it in the table)")
            comps.append(Component(name, "Q", [node(t) for t in tk[1:4]], None, params={"polarity": pol, "beta": kv.get("bf") or 100.0}, note=mname))
        elif k0 == "M" and len(tk) >= 6:
            mname = tk[5]; typ, kv = models.get(mname.lower(), (None, {})); pol = "P" if (typ == "PMOS" or (typ is None and "pmos" in mname.lower())) else "N"
            opts = {k.lower(): parse_value(v) for k, v in re.findall(r"(\w+)\s*=\s*([^\s]+)", " ".join(tk[6:]))}
            kp = (kv.get("kp") or 2e-3) * ((opts["w"] / opts["l"]) if opts.get("w") and opts.get("l") else 1.0)
            if typ is None: warns.append(f"{name}: model '{mname}' has no .model card - assumed Kp=2 mA/V^2, Vth=1 V")
            comps.append(Component(name, "M", [node(t) for t in (tk[1], tk[2], tk[3])], None,
                                   params={"type": pol, "kp": kp, "vth": abs(kv.get("vto") if kv.get("vto") is not None else 1.0), "lam": kv.get("lambda") or 0.0}, note=mname))
        elif k0 == "E" and len(tk) >= 6 and "=" not in tk[3] and parse_value(tk[5]) is not None:
            comps.append(Component(name, "E", [node(t) for t in tk[1:5]], parse_value(tk[5])))
        else:
            cnt = {"Q": 3, "J": 3, "M": 4, "X": len(tk) - 2}.get(k0, 2)
            comps.append(Component(name, "UNKNOWN", [node(n) for n in tk[1:1 + max(cnt, 1)]], None, note=f"{name}: {k0}-type device is not supported yet"))
            warns.append(f"{name}: not supported yet ({ {'Q': 'BJT', 'M': 'MOSFET', 'X': 'subcircuit/op-amp', 'J': 'JFET'}.get(k0, 'device') })")
    return _finish(comps, names, warns, freq)

# ------------------------------------------------ KiCad .net (S-expression) --------------------------------------------
def _sexp(text):
    toks = re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()]+', text); pos = 0
    def rd():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t == "(":
            out = []
            while toks[pos] != ")": out.append(rd())
            pos += 1; return out
        return t[1:-1].replace('\\"', '"') if t.startswith('"') else t
    return rd()

def _find(node, key): return [c for c in node if isinstance(c, list) and c and c[0] == key]
def _val(node, key):
    r = _find(node, key); return r[0][1] if r and len(r[0]) > 1 else None

def parse_kicad_net(text):
    root = _sexp(text)
    if not isinstance(root, list) or not root or root[0] != "export": raise ValueError("Not a KiCad netlist (.net)")
    info = {}
    for comp in (_find(_find(root, "components")[0], "comp") if _find(root, "components") else []):
        props = {_val(p, "name"): _val(p, "value") for p in _find(comp, "property")}
        ls = _find(comp, "libsource"); info[_val(comp, "ref")] = {"value": _val(comp, "value") or "", "part": (_val(ls[0], "part") if ls else "") or "", "props": props}
    pins, names, ids, warns, freq, nid = {}, {}, {}, [], None, 0
    for net in (_find(_find(root, "nets")[0], "net") if _find(root, "nets") else []):
        nm = _val(net, "name") or ""
        if _GND.search(nm): i = 0; names[0] = nm
        else: nid += 1; i = nid; names[i] = nm
        for nd in _find(net, "node"): pins.setdefault(_val(nd, "ref"), {})[str(_val(nd, "pin"))] = (i, _val(nd, "pinfunction") or "")
    comps = []
    for ref in sorted(info, key=lambda r: (re.sub(r"\d", "", r), int(re.sub(r"\D", "", r) or 0))):
        d, pp = info[ref], pins.get(ref, {})
        if not pp: continue
        k0, part = ref[0].upper(), d["part"].upper()
        def free():
            nonlocal nid
            nid += 1; return nid
        get = lambda p: pp[p][0] if p in pp else free()
        if k0 in "RCL" and not part.startswith(("SW", "CONN")) and ref[:1].isalpha():
            v = parse_value(d["value"], k0); comps.append(Component(ref, k0, [get("1"), get("2")], v))
            if v is None: warns.append(f"{ref}: value '{d['value']}' not understood")
        elif k0 == "Q" and len(pp) == 3 and ("NPN" in part or "PNP" in part or "NMOS" in part or "PMOS" in part or _guess_pol(d["value"])):
            mos = "MOS" in part; want = "DGS" if mos else "CBE"; sfx = re.search(r"_([A-Z]{3})$", part)
            fn = {p_: (f or "").upper() for p_, (_, f) in pp.items()}
            if set(fn.values()) != set(want):                                    # no usable pin names: fall back to the symbol's suffix, e.g. Q_NPN_BCE
                if not sfx or set(sfx.group(1)) != set(want): comps.append(Component(ref, "UNKNOWN", [pp[p_][0] for p_ in sorted(pp)], None, note=f"{ref}: cannot tell which pin is which")); warns.append(f"{ref}: pin roles unknown"); continue
                fn = dict(zip(sorted(pp), sfx.group(1)))
            pin = {role: p_ for p_, role in fn.items()}
            if mos: comps.append(Component(ref, "M", [pp[pin["D"]][0], pp[pin["G"]][0], pp[pin["S"]][0]], None, params={"type": "P" if "PMOS" in part else "N", "kp": 2e-3, "vth": 1.0, "lam": 0.0}, note=d["value"]))
            else:
                pol = "PNP" if "PNP" in part else "NPN" if "NPN" in part else _guess_pol(d["value"]) or "NPN"
                comps.append(Component(ref, "Q", [pp[pin["C"]][0], pp[pin["B"]][0], pp[pin["E"]][0]], None, params={"polarity": pol, "beta": 100.0}, note=d["value"]))
            warns.append(f"{ref}: transistor parameters are defaults (beta 100 / Kp 2 mA/V^2, Vth 1 V) - edit them in the table")
        elif k0 == "U" and {"+", "-"} <= {(f or "") for _, f in pp.values()} and any((f or "") in ("~", "OUT", "OUTPUT", "Out") for _, f in pp.values()):
            role = {f: p_ for p_, (_, f) in pp.items()}; outp = next(role[f] for f in ("~", "OUT", "OUTPUT", "Out") if f in role)
            comps.append(Component(ref, "OA", [pp[outp][0], pp[role["+"]][0], pp[role["-"]][0]], None, note=d["value"]))
            warns.append(f"{ref}: modelled as an IDEAL op-amp (infinite gain, no rails) - set vpos/vneg in its parameters for clipping")
        elif k0 in "VI" and len(pp) == 2:
            pos = next((p for p, (_, fn) in pp.items() if fn == "+"), "1"); neg = next(p for p in pp if p != pos)
            v, ph, fq, note = _source(d["props"].get("Sim.Params") or d["value"]); freq = fq or freq
            comps.append(Component(ref, k0, [pp[pos][0], pp[neg][0]], v, note=note, phase=ph))
            if v is None: warns.append(f"{ref}: source value '{d['value']}' not understood")
        elif k0 == "D" and len(pp) == 2:
            a = next((p for p, (_, fn) in pp.items() if fn.upper() == "A"), "2"); k = next(p for p in pp if p != a)
            comps.append(Component(ref, "LED" if "LED" in part else "D", [pp[a][0], pp[k][0]], None))
        else:
            comps.append(Component(ref, "UNKNOWN", [pp[p][0] for p in sorted(pp)], None, note=f"{ref} ({d['part'] or d['value']}) is not supported yet"))
            warns.append(f"{ref}: {d['part'] or d['value']} not supported yet")
    return _finish(comps, names, warns, freq)

def load(text):
    t = text.lstrip()
    if t.startswith("(kicad_sch") or t.startswith("(kicad_pcb"): raise ValueError("That is a KiCad schematic/board file. In KiCad use File > Export > Netlist (SPICE .cir, or KiCad .net) and upload that instead.")
    return parse_kicad_net(text) if t.startswith("(export") else parse_spice(text)
