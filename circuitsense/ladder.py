"""Ladder-logic diagram image -> structured ladder (rails, rungs, contacts NO/NC, coils, addresses)."""
import re, cv2, numpy as np
from . import ocr
from .vision import _clusters

def _cc(m):
    return cv2.connectedComponentsWithStats(m, 8)[2][1:]

def _elem(c):
    on = c > 0; r_ = np.where(on.any(1))[0]
    if len(r_) < 6: return "UNKNOWN"
    on = on[r_[0]:r_[-1] + 1]; h, w = on.shape                                          # trim to the symbol itself
    half = on[:, :max(2, w // 2)]; mid = h // 2; bw = max(2, int(.12 * h))              # ignore the rung wire's own rows
    rows = [r for r in np.where(half.any(1))[0] if abs(r - mid) > bw]
    if w < 4 or len(rows) < 6: return "UNKNOWN"
    if np.array([np.argmax(half[r]) for r in rows]).std() > 1.8: return "COIL"     # arcs/circle bend, contact bars are straight
    bars = [b for b in _clusters(on.sum(0), .3 * h) if b[1] - b[0] <= max(8, .3 * w)]
    if len(bars) != 2: return "UNKNOWN"
    gap = on[:, bars[0][1] + 2:bars[1][0] - 1]
    return "NC" if gap.size and gap.any(0).mean() > .3 else "NO"                     # slash between the bars = normally closed

def analyze_ladder(img, max_dim=1200):
    m = max(img.shape[:2]); s = min(1.0, max_dim / m) if m > 900 else 900 / m
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    g = img.min(2) if img.ndim == 3 else img
    ink = cv2.threshold(cv2.divide(g, cv2.GaussianBlur(g, (0, 0), 25), scale=255), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lab_all = ocr.read_all(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    for txt, (x0, y0, x1, y1) in lab_all:
        if len(re.findall(r"[A-Za-z]", txt)) >= 2 and re.fullmatch(r"[%\w ./\-]{2,40}", txt): ink[y0:y1 + 1, x0:x1 + 1] = 0
    H_, W_ = ink.shape; t = 5; out = {"kind": "ladder", "elements": [], "warnings": [], "scale": s}
    Lr = int(.4 * H_)
    rails = sorted((x, x + w, y, y + h) for x, y, w, h, _ in _cc(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((Lr, 1), np.uint8))) if h >= Lr)
    if len(rails) < 2:
        out["warnings"].append("could not find two vertical power rails"); return out
    xl, xr = rails[0][1], rails[-1][0]
    out["rung_numbers"] = sorted((int(t), (y0 + y1) // 2) for t, (x0, y0, x1, y1) in lab_all if t.isdigit() and (x0 + x1) / 2 < rails[0][0])
    out["height"] = H_
    ink[:, :max(0, rails[0][0] - 1)] = 0; ink[:, rails[-1][1] + 2:] = 0     # rung numbers / margin notes are not logic
    for x0, x1, y0, y1 in (rails[0], rails[-1]): ink[y0:y1 + 1, max(0, x0 - 1):x1 + 2] = 0
    Lh, Lv = max(20, int(.03 * W_)), max(15, int(.02 * H_))
    segs = sorted(((x, y + h // 2, x + w - 1) for x, y, w, h, _ in _cc(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, Lh), np.uint8)))), key=lambda q: q[1])
    vs = [(x + w // 2, y, y + h - 1) for x, y, w, h, _ in _cc(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((Lv, 1), np.uint8)))]
    lines = []
    for sg in segs:
        if lines and abs(sg[1] - lines[-1][0][1]) <= t: lines[-1].append(sg)
        else: lines.append([sg])
    lines = [sorted(l) for l in lines]; flat = [sg for l in lines for sg in l]; idx = {sg: i for i, sg in enumerate(flat)}
    n = len(flat); P = list(range(n + 2)); LR, RR = n, n + 1
    def f(a):
        while P[a] != a: P[a] = P[P[a]]; a = P[a]
        return a
    def cover(x, y): return next((idx[q] for q in flat if abs(q[1] - y) <= t and q[0] - t <= x <= q[2] + t), None)
    joint = set()
    for x, y0, y1 in vs:                                         # connectors have wire at BOTH ends; contact bars don't
        a, b = cover(x, y0), cover(x, y1)
        if a is not None and b is not None and a != b:
            P[f(a)] = f(b); joint |= {(a, "L") if abs(x - flat[a][0]) <= t else (a, "M"), (a, "R") if abs(x - flat[a][2]) <= t else (a, "M"),
                                      (b, "L") if abs(x - flat[b][0]) <= t else (b, "M"), (b, "R") if abs(x - flat[b][2]) <= t else (b, "M")}
    for i, q in enumerate(flat):
        if abs(q[0] - xl) <= t: P[f(i)] = f(LR)
        if abs(q[2] - xr) <= t: P[f(i)] = f(RR)
    hh = max(20, int(.06 * H_))
    out["texts"] = [t for t, _ in lab_all]
    if ocr.error: out["warnings"].append(f"OCR NOT RUNNING ({ocr.error}) - addresses were not read; type them in the table")
    gaps = []                                                    # (line y, left node idx, right node idx, x0, x1)
    for l in lines:
        y = int(np.mean([q[1] for q in l])); i0, i1 = idx[l[0]], idx[l[-1]]
        if l[0][0] - xl > 2 * t and (i0, "L") not in joint and l[0][0] - xl <= 120: gaps.append((y, LR, i0, xl, l[0][0]))
        for a, b in zip(l, l[1:]):
            if b[0] - a[2] > 2 * t: gaps.append((y, idx[a], idx[b], a[2], b[0]))
        if xr - l[-1][2] > 2 * t and (i1, "R") not in joint and xr - l[-1][2] <= 120: gaps.append((y, i1, RR, l[-1][2], xr))
    labels, flavor = [], {}
    for txt, (x0, y0, x1, y1) in lab_all:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        gi = next((k for k, gp in enumerate(gaps) if gp[3] <= cx <= gp[4] and abs(gp[0] - cy) <= hh), None)
        if any(y0 - 3 <= gp[0] <= y1 + 3 for gp in gaps) and gi is not None:            # text sitting ON a rung, in a gap: coil flavour letter
            if txt.upper() in ("L", "S", "SET"): flavor[gi] = "latch"; ink[y0:y1 + 1, x0:x1 + 1] = 0
            elif txt.upper() in ("U", "R", "RST"): flavor[gi] = "unlatch"; ink[y0:y1 + 1, x0:x1 + 1] = 0
            # anything else here is the OCR misreading the symbol itself ("II", "1 1"): ignore it, never erase
        elif re.search(r"[A-Za-z]", txt) and re.fullmatch(r"[%\w ./\-]{1,40}", txt) and not any(y0 - 3 <= q[1] <= y1 + 3 for q in flat):
            labels.append([txt, x0, y0, x1, y1]); ink[y0:y1 + 1, x0:x1 + 1] = 0
    labels.sort(key=lambda q: (q[1] + q[3]) / 2 // 1)
    merged = []
    for q in sorted(labels, key=lambda q: q[2]):                 # "Low Sensor" / "Switch" on two lines -> one name
        for m_ in merged:
            if abs((q[1] + q[3]) - (m_[1] + m_[3])) / 2 < .35 * max(q[3] - q[1], m_[3] - m_[1]) and 0 < q[2] - m_[4] < .5 * (q[4] - q[2]):
                m_[0] += " " + q[0]; m_[1], m_[3], m_[4] = min(m_[1], q[1]), max(m_[3], q[3]), q[4]; break
        else: merged.append(list(q))
    labels = [((ocr.norm_addr(q[0]) if len(q[0]) <= 6 else q[0]), (q[1] + q[3]) / 2, (q[2] + q[4]) / 2) for q in merged]
    ids = {}
    def node(i):
        r = f(i); return 1 if r == f(LR) else 2 if r == f(RR) else ids.setdefault(r, len(ids) + 3)
    for k, (y, a, b, x0, x1) in enumerate(sorted(gaps, key=lambda gp: (gp[0], gp[3]))):
        kind = _elem(ink[max(0, y - hh):y + hh, max(0, x0 - 4):x1 + 5])   # bars sit right at the wire ends
        near = sorted(((abs(cy - y) + (0 if cy < y else 30), txt) for txt, cx, cy in labels if x0 - hh <= cx <= x1 + hh and abs(cy - y) <= 3 * hh))
        e = {"id": f"E{k + 1}", "type": kind, "addr": near[0][1] if near else None, "coil": "normal", "nodes": [node(a), node(b)],
             "bbox": [int(x0), int(y - hh), int(x1), int(y + hh)], "x": int((x0 + x1) / 2), "y": int(y)}
        gi = next((j for j, gp in enumerate(gaps) if gp == (y, a, b, x0, x1)), None)
        if kind == "COIL" and flavor.get(gi): e["coil"] = flavor[gi]
        if kind == "UNKNOWN": out["warnings"].append(f"{e['id']}: unrecognised element")
        if not e["addr"]: out["warnings"].append(f"{e['id']}: no address read")
        out["elements"].append(e)
    return out
