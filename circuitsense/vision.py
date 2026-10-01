"""Classical CV: image -> netlist. Axis-aligned schematics only (v1).
Wires = long horizontal/vertical runs. Everything left over touching a wire = a component."""
import cv2, numpy as np
from .model import Component, Circuit
from . import ocr, ai_fallback

def _clusters(prof, thr):
    idx = np.where(prof >= thr)[0]
    if not len(idx): return []
    out, s = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b - a > 2: out.append((s, a)); s = b
    out.append((s, idx[-1])); return out

def classify(crop, sides, rel=0.5):
    on = crop > 0; h, w = on.shape
    if len(sides) == 1 and "T" in sides:                       # ground: stacked bars
        return ("GND", 0) if len(_clusters(on.sum(1), 0.3 * w)) >= 2 else ("UNKNOWN", 0)
    if not ({"L", "R"} <= sides or {"T", "B"} <= sides): return "UNKNOWN", 0
    if {"T", "B"} <= sides: on = on.T; h, w = on.shape         # put axis horizontal
    if w >= 1.2 * h and (rel > 0.8 or rel < 0.2): return "L", 0   # coil bumps sit to one side of the wire
    col = on.sum(0); bars = [c for c in _clusters(col, 0.3 * h) if c[1] - c[0] <= max(8, 0.15 * w)]  # plates are thin
    empty = all(len(col[a[1] + 1:b[0]]) and (col[a[1] + 1:b[0]] > 0).mean() < .2 for a, b in zip(bars, bars[1:]))
    if len(bars) >= 2 and (len(bars) == 2 or empty):     # >2 bars only counts as a multi-cell battery if the gaps are empty
        (a0, a1), (b0, b1) = bars[:2]; gap = on[:, a1 + 1:b0]; gc = col[a1 + 1:b0]
        ha, hb = col[a0:a1 + 1].max(), col[b0:b1 + 1].max()
        if len(bars) == 2 and len(gc) and (gc > 0).mean() > 0.8:
            if min(ha, hb) >= 0.6 * h and gap[int(.4 * h):int(.6 * h) + 1].any(0).mean() > 0.15:  # triangle + bar
                ys = np.arange(h)[:, None]; th = max(1, gap.shape[1] // 3)
                sp = lambda g: (np.abs(ys - h / 2) * g).sum() / max(1, g.sum())   # wide side = anode
                return ("D" if min(ha, hb) >= 0.8 * h else "LED"), (0 if sp(gap[:, :th]) > sp(gap[:, -th:]) else 1)
            return "R", 0                                       # IEC box
        if min(ha, hb) / max(ha, hb) > 0.8: return ("C", 0) if len(bars) == 2 else ("UNKNOWN", 0)
        return "V", (0 if ha > hb else 1)                      # long plate = +
    if w >= 1.2 * h: return "R", 0                              # zigzag (vertex columns look like bars)
    return "UNKNOWN", 0

def source(inner, vertical):
    """Circle symbol: arrow shaft through the centre = current source, empty centre (+/- glyphs) = voltage source."""
    on = (inner > 0).T if vertical else inner > 0
    h, w = on.shape; t = max(1, w // 3)
    shaft = on[int(.4 * h):int(.6 * h) + 1, int(.35 * w):int(.65 * w) + 1].any(0).mean() > .8
    return shaft, int(on[:, :t].sum()), int(on[:, -t:].sum())     # more ink at an end = arrow head / plus sign

def analyze(img, wire_frac=0.06, max_dim=1200, join_crossings=False):
    s = min(1.0, max_dim / max(img.shape[:2]))
    if s < 1: img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = img.min(2) if img.ndim == 3 else img                     # min channel: coloured ink stays dark
    flat = cv2.divide(g, cv2.GaussianBlur(g, (0, 0), 25), scale=255)
    ink = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    labels = []                                                  # (text, box, glyph count) - then erased so they can't merge into parts
    for txt, (x0, y0, x1, y1) in ocr.read_all(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)):
        if ocr.is_label(txt):
            gl = cv2.connectedComponents(cv2.dilate(ink[y0:y1 + 1, x0:x1 + 1], np.ones((3, 3), np.uint8)))[0] - 1
            labels.append((txt, (x0, y0, x1, y1), gl))
    for _, (x0, y0, x1, y1), _ in labels: ink[y0 + 1:y1, x0 + 1:x1] = 0
    L = max(15, int(wire_frac * max(ink.shape))); t = max(4, L // 8)
    hm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    vm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    resid = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(hm | vm, np.ones((3, 3), np.uint8))))
    blur, inkd = cv2.GaussianBlur(ink, (5, 5), 0), cv2.dilate(ink, np.ones((5, 5), np.uint8))
    def hough(p2, r0, r1):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 12, param1=100, param2=p2, minRadius=r0, maxRadius=r1)
        return [] if c is None else [tuple(map(float, q)) for q in c[0]]
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = float(np.percentile(dt[hm > 0], 90)) if hm.any() else 1.0
    def runs(m, k):
        return [(k, x, y, x + w - 1, y + h - 1) for x, y, w, h, _ in cv2.connectedComponentsWithStats(m, 8)[2][1:]]
    R = runs(hm, "h") + runs(vm, "v")
    near = lambda j, x, y, d: R[j][1] - d <= x <= R[j][3] + d and R[j][2] - d <= y <= R[j][4] + d

    def side(r, b):
        k, rx0, ry0, rx1, ry1 = r; x0, y0, x1, y1 = b
        if rx1 < x0 - t or rx0 > x1 + t or ry1 < y0 - t or ry0 > y1 + t: return None
        if rx0 >= x0 - t and rx1 <= x1 + t and ry0 >= y0 - t and ry1 <= y1 + t: return "own"
        if k == "h":
            yc = (ry0 + ry1) / 2
            if abs(rx1 - x0) <= t and y0 - t <= yc <= y1 + t: return "L"
            if abs(rx0 - x1) <= t and y0 - t <= yc <= y1 + t: return "R"
            if abs(yc - y0) <= t: return "T"
            if abs(yc - y1) <= t: return "B"
        else:
            xc = (rx0 + rx1) / 2
            if abs(ry1 - y0) <= t and x0 - t <= xc <= x1 + t: return "T"
            if abs(ry0 - y1) <= t and x0 - t <= xc <= x1 + t: return "B"
            if abs(xc - x0) <= t: return "L"
            if abs(xc - x1) <= t: return "R"

    found, owned, junc = [], set(), []
    jc = [(x, y, r) for x, y, r in hough(14, 4, max(6, int(.22 * L))) if sum(near(j, x, y, r + t + 6) for j in range(len(R))) >= 2]
    for x, y, r in jc: cv2.circle(resid, (int(x), int(y)), int(r) + 7, 0, -1)       # open-circle node markers
    srcs = []
    for x, y, r in hough(18, max(8, int(.3 * L)), int(1.1 * L)):                       # circle symbols = sources
        if any((x - a) ** 2 + (y - b) ** 2 < r * r for a, b, _ in srcs): continue
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        px = np.clip((x + r * np.cos(ang)).astype(int), 0, ink.shape[1] - 1); py = np.clip((y + r * np.sin(ang)).astype(int), 0, ink.shape[0] - 1)
        if (inkd[py, px] > 0).mean() < .8: continue
        box = (int(x - r), int(y - r), int(x + r), int(y + r)); sd = {}; own = set()
        for j, rr in enumerate(R):
            q = side(rr, box)
            if q == "own": own.add(j)
            elif q: sd.setdefault(q, j)
        if {"L", "R"} <= set(sd) or {"T", "B"} <= set(sd):
            srcs.append((x, y, r)); owned |= own; cv2.circle(resid, (int(x), int(y)), int(r + 2 * stroke) + 3, 0, -1); found.append((box, sd, (x, y, r)))   # wire runs end at the ring's inner edge; erase the whole ring

    kk = np.ones((max(5, int(4 * stroke) + 1) | 1,) * 2, np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(cv2.dilate(resid, kk), 8)
    boxes = []
    for i in range(1, n):
        ys, xs = np.where((lab == i) & (resid > 0))
        if len(xs) >= 15: boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), len(xs)])
    for a in range(len(R)):                                    # inline parts sit in gaps of collinear wires
        for c in range(a + 1, len(R)):
            (ka, ax0, ay0, ax1, ay1), (kc, cx0, cy0, cx1, cy1) = R[a], R[c]
            if ka != kc: continue
            if ka == "h" and abs(ay0 - cy0) <= t and (ax1 < cx0 or cx1 < ax0):
                g0, g1 = (ax1, cx0) if ax1 < cx0 else (cx1, ax0); yy = (ay0 + ay1) / 2
                box = (g0 - t, yy - .3 * L, g1 + t, yy + .3 * L)
            elif ka == "v" and abs(ax0 - cx0) <= t and (ay1 < cy0 or cy1 < ay0):
                g0, g1 = (ay1, cy0) if ay1 < cy0 else (cy1, ay0); xx = (ax0 + ax1) / 2
                box = (xx - .3 * L, g0 - t, xx + .3 * L, g1 + t)
            else: continue
            if (box[2] - box[0]) * (box[3] - box[1]) > .9 * L * L: continue
            inn = [q for q in boxes if box[0] <= (q[0] + q[2]) / 2 <= box[2] and box[1] <= (q[1] + q[3]) / 2 <= box[3]]
            if len(inn) > 1:
                m = [min(q[0] for q in inn), min(q[1] for q in inn), max(q[2] for q in inn), max(q[3] for q in inn), sum(q[4] for q in inn)]
                boxes = [q for q in boxes if not any(q is z for z in inn)] + [m]
    for bx in boxes:
        b = tuple(bx[:4]); sd = {}
        for j, r in enumerate(R):
            q = side(r, b)
            if q == "own": owned.add(j)
            elif q: sd.setdefault(q, j)
        bw, bh = b[2] - b[0], b[3] - b[1]
        if len(sd) >= 2 and max(bw, bh) <= .35 * L and .6 <= (bw + 1) / (bh + 1) <= 1.6: junc.append(sd)
        elif sd: found.append((b, sd, None))

    P = list(range(len(R)))
    def f(a):
        while P[a] != a: P[a] = P[P[a]]; a = P[a]
        return a
    live = [j for j in range(len(R)) if j not in owned]
    for a in live:
        for b in live:
            if a >= b: continue
            (ka, ax0, ay0, ax1, ay1), (kb, bx0, by0, bx1, by1) = R[a], R[b]
            if ka == kb:
                if ka == "h" and abs(ay0 - by0) <= t and ax0 - t <= bx1 and bx0 - t <= ax1: P[f(a)] = f(b)
                if ka == "v" and abs(ax0 - bx0) <= t and ay0 - t <= by1 and by0 - t <= ay1: P[f(a)] = f(b)
                continue
            h, v = (R[a], R[b]) if ka == "h" else (R[b], R[a])
            xc, yc = (v[1] + v[3]) / 2, (h[2] + h[4]) / 2
            if not (h[1] - t <= xc <= h[3] + t and v[2] - t <= yc <= v[4] + t): continue
            ends = min(abs(xc - h[1]), abs(xc - h[3]), abs(yc - v[2]), abs(yc - v[4])) <= t
            dot = dt[int(yc), int(xc)] >= 1.8 * stroke        # filled dot = junction; bare crossing = no connection
            if ends or dot or join_crossings: P[f(a)] = f(b)   # bare crossings: convention differs between textbooks
    for sd in junc:
        js = list(sd.values())
        for j in js[1:]: P[f(j)] = f(js[0])
    for x, y, r in jc:
        hit = [j for j in live if near(j, x, y, r + t + 6)]
        for j in hit[1:]: P[f(j)] = f(hit[0])

    lv = []                                                      # (cx, cy, value, unit) for value labels
    for txt, (x0, y0, x1, y1), gl in labels:
        val, u = ocr.split(txt, gl)
        if val is not None: lv.append(((x0 + x1) / 2, (y0 + y1) / 2, val, u))
    ids, cnt, comps, warns = {}, {}, [], []
    nid = lambda j: ids.setdefault(f(j), len(ids) + 1)
    for b, sd, cf in sorted(found, key=lambda e: (e[0][1], e[0][0])):
        S = set(sd); crop = ink[b[1]:b[3] + 1, b[0]:b[2] + 1]
        a_s, b_s = ("T", "B") if {"T", "B"} <= S and not {"L", "R"} <= S else ("L", "R")
        if a_s not in S: a_s = next(iter(S)); b_s = None
        ends = [nid(sd[a_s])] + ([nid(sd[b_s])] if b_s in sd else [])
        if cf:
            x, y, r = cf; m = np.zeros(ink.shape, np.uint8); cv2.circle(m, (int(x), int(y)), int(.72 * r), 255, -1)
            shaft, lft, rgt = source(cv2.bitwise_and(ink, m)[b[1]:b[3] + 1, b[0]:b[2] + 1], a_s == "T")
            us = sorted((((lx - x) ** 2 + (ly - y) ** 2, u) for lx, ly, _, u in lv if u in ("A", "V") and (lx - x) ** 2 + (ly - y) ** 2 < (4 * L) ** 2))
            kind = us[0][1].replace("A", "I") if us else ("I" if shaft else "V")   # printed unit beats geometry
            o = (0 if rgt > lft else 1) if kind == "I" else (0 if lft > rgt else 1)  # I: head end = b ; V: plus end = first node
        else:
            rel = 0.5
            if b_s in S:
                r0 = R[sd[a_s]]
                rel = ((r0[2] + r0[4]) / 2 - b[1]) / max(1, b[3] - b[1]) if a_s == "L" else ((r0[1] + r0[3]) / 2 - b[0]) / max(1, b[2] - b[0])
            kind, o = classify(crop, S, rel)
        if kind == "GND": ends = [nid(sd["T"])]
        elif kind in ("V", "D", "LED", "I") and o == 1: ends = ends[::-1]
        elif kind != "UNKNOWN" and len(ends) != 2: kind = "UNKNOWN"
        cnt[kind] = cnt.get(kind, 0) + 1
        c = Component(f"{kind}{cnt[kind]}", kind, ends, None, [int(v) for v in b])
        if kind == "UNKNOWN":
            c.note = "unrecognised symbol - send crop to AI fallback or fix manually"; warns.append(c.id)
        comps.append(c)
    unit_of = {"R": "Ω", "I": "A", "V": "V"}                    # each label goes to the nearest part that takes that unit
    pairs = sorted(((((lx - (c.bbox[0] + c.bbox[2]) / 2) ** 2 + (ly - (c.bbox[1] + c.bbox[3]) / 2) ** 2) ** .5, li, ci)
                    for li, (lx, ly, _, u) in enumerate(lv) for ci, c in enumerate(comps) if unit_of.get(c.kind) == u))
    du, dc = set(), set()
    for d, li, ci in pairs:
        if d < 4 * L and li not in du and ci not in dc: comps[ci].value = lv[li][2]; du.add(li); dc.add(ci)
    for c in comps:                                              # AI only identifies what CV could not
        if c.kind != "UNKNOWN": continue
        x0, y0, x1, y1 = c.bbox; pad = 6
        ok, png = cv2.imencode(".png", g[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad])
        r = ai_fallback.resolve(png.tobytes()) if ok else None
        if r and r["kind"] != "UNKNOWN" and len(c.nodes) == 2:
            c.kind, c.value, c.note = r["kind"], r.get("value"), "identified by AI fallback"
            if r.get("reversed"): c.nodes = c.nodes[::-1]
            warns.remove(c.id)
    msgs = [f"{w}: unresolved" for w in warns]
    if ocr.error: msgs.insert(0, f"OCR NOT RUNNING ({ocr.error}) - labels/values were not read; fix: pip install rapidocr-onnxruntime")
    return Circuit(comps, msgs, s)
