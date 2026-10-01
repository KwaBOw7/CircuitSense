import sys, pytest; import numpy as np; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import gen
from circuitsense.model import Circuit, Component as K
from circuitsense.solver import solve, whatif
from circuitsense.vision import analyze
from circuitsense import ladder_sim as L, wiring, qa
from circuitsense.ladder import analyze_ladder
ocr_ok = pytest.mark.skipif(pytest.importorskip("rapidocr_onnxruntime", reason="OCR not installed") is None, reason="")

# ---- solver: independent textbook answers -------------------------------------------------
def test_divider():
    c = Circuit([K("V1", "V", [1, 3], 10), K("R1", "R", [1, 2], 1000), K("R2", "R", [2, 3], 1000), K("G", "GND", [3])])
    assert solve(c)["node_v"][2] == pytest.approx(5.0)

def test_current_source_into_resistor():
    c = Circuit([K("I1", "I", [3, 1], 1.0), K("R1", "R", [1, 3], 2.0), K("G", "GND", [3])])
    assert solve(c)["node_v"][1] == pytest.approx(2.0)

def test_led_forward_current():
    c = Circuit([K("V1", "V", [1, 3], 12), K("R1", "R", [1, 2], 470), K("LED1", "LED", [2, 3]), K("G", "GND", [3])])
    r = solve(c)["parts"]; assert r["LED1"]["i"] == pytest.approx(10 / 470) and r["LED1"]["on"]

def test_reverse_diode_blocks():
    c = Circuit([K("V1", "V", [1, 3], 5), K("R1", "R", [1, 2], 100), K("D1", "D", [3, 2]), K("G", "GND", [3])])
    assert solve(c)["parts"]["D1"]["i"] == 0.0

def test_undefined_circuit_is_rejected():
    c = Circuit([K("I1", "I", [3, 1], 1.0), K("R1", "R", [4, 3], 2.0), K("G", "GND", [3])])
    with pytest.raises(ValueError, match="no path to ground"): solve(c)

def test_whatif_changes_only_what_it_should():
    c = Circuit([K("V1", "V", [1, 3], 10), K("R1", "R", [1, 2], 1000), K("R2", "R", [2, 3], 1000), K("G", "GND", [3])])
    w = whatif(c, {"R2": 3000}); assert w["after"]["node_v"][2] == pytest.approx(7.5) and c.components[2].value == 1000
    with pytest.raises(ValueError): whatif(c, {"R9": 1})

# ---- vision: the drawing -> netlist -> solved -----------------------------------------------
def test_schematic_pipeline():
    c = analyze(gen.circuit_image(), join_crossings=True)
    assert sorted(x.kind for x in c.components) == ["GND", "I", "I", "R", "R", "R", "V"] and not c.warnings or all("OCR" in w for w in c.warnings)
    vals = {"R1": 10, "R2": 2, "R3": 4, "V1": 2, "I1": 2, "I2": 7}
    for x in c.components: x.value = vals.get(x.id)
    nv = solve(c)["node_v"]; assert sorted(round(v, 3) for v in nv.values()) == [-7.333, -5.333, 0.0]   # hand-checked with nodal analysis

def test_crossings_flag_matters():
    c = analyze(gen.circuit_image(), join_crossings=False)
    for x in c.components: x.value = 1.0
    with pytest.raises(ValueError): solve(c)

@ocr_ok
def test_ocr_reads_values():
    c = analyze(gen.circuit_image(), join_crossings=True)
    assert sorted(x.value for x in c.components if x.value is not None) == [2, 2, 2, 4, 7, 10]

# ---- ladder ---------------------------------------------------------------------------------
@ocr_ok
def test_ladder_seal_in():
    lad = analyze_ladder(gen.ladder_image()); assert not lad["warnings"]
    assert sorted(e["type"] for e in lad["elements"]) == ["COIL", "COIL", "NC", "NO", "NO", "NO"]
    s = L.scan(lad, {}, {"X0": True}); assert s["Y0"] and s["Y1"]
    s = L.scan(lad, s, {"X0": False}); assert s["Y0"], "seal-in must hold after START is released"
    s = L.scan(lad, s, {"X1": True}); assert not s["Y0"] and not s["Y1"]
    assert "Y0 = X0·¬X1 + Y0·¬X1" in L.equations(lad)
    assert L.rung_text(lad)[0].startswith("Rung 1:") and L.rung_text(lad)[1].startswith("Rung 2:")

def test_latch_unlatch():
    lad = {"elements": [{"id": "a", "type": "NO", "addr": "S", "nodes": [1, 3], "coil": "normal", "y": 1, "x": 1},
                        {"id": "b", "type": "COIL", "addr": "M", "nodes": [3, 2], "coil": "latch", "y": 1, "x": 2},
                        {"id": "c", "type": "NO", "addr": "R", "nodes": [1, 4], "coil": "normal", "y": 2, "x": 1},
                        {"id": "d", "type": "COIL", "addr": "M", "nodes": [4, 2], "coil": "unlatch", "y": 2, "x": 2}]}
    s = L.scan(lad, {}, {"S": True}); s = L.scan(lad, s, {"S": False}); assert s["M"]
    assert not L.scan(lad, s, {"R": True})["M"]

# ---- wiring + Q&A ---------------------------------------------------------------------------
def test_wiring_flags_short_and_continuity():
    c = Circuit([K("V1", "V", [1, 1], 5), K("R1", "R", [1, 2], 10)])
    r = wiring.analyze(c, "R1.1", "V1.+"); assert any("SHORT" in i for i in r["issues"]) and r["continuity"]["connected"]

def test_qa_ai_only_plans_solver_computes():
    c = Circuit([K("V1", "V", [1, 3], 10), K("R1", "R", [1, 2], 1000), K("R2", "R", [2, 3], 1000), K("G", "GND", [3])]).to_dict()
    calls = []
    def fake(text, as_json=False):
        calls.append(text); return {"answer": "", "changes": {"R2": 3000}} if as_json else "ok"
    r = qa.ask(c, "what if R2 is 3k?", gen=fake)
    assert r["whatif"]["after"]["node_v"][2] == pytest.approx(7.5) and "7.5" in calls[1]

def test_no_reference_is_a_clean_error():
    with pytest.raises(ValueError, match="ground"): solve(Circuit([K("R1", "R", [1, 2], 10.0)]))

@ocr_ok
def test_explain_endpoint_receives_the_parsed_ladder():          # regression: the form field used to be dropped silently
    import json
    from fastapi.testclient import TestClient
    from circuitsense.api import app
    c = TestClient(app); img = __import__("cv2").imencode(".png", gen.ladder_image())[1].tobytes(); f = {"file": ("l.png", img, "image/png")}
    lad = c.post("/analyze?kind=ladder", files=f).json()
    r = c.post("/ladder/explain", files=f, data={"ladder": json.dumps(lad)}).json()
    assert r["verified"] and r["deterministic"][0].startswith("Rung 1:")

# ---- inverse: find unknown values from measurements ------------------------------------------
from circuitsense import inverse
def _div(**v):
    return Circuit([K("V1", "V", [1, 3], v.get("V1", 10)), K("R1", "R", [1, 2], v.get("R1", 1000)), K("R2", "R", [2, 3], v.get("R2", 1000)), K("G", "GND", [3])])

def test_find_one_resistor():
    f = inverse.find(_div(R1=None), [{"kind": "v", "of": "R2", "value": 2.5}])          # 10 V, R2 = 1k, 2.5 V across R2  ->  R1 = 3k
    assert f["values"]["R1"] == pytest.approx(3000, rel=1e-5) and f["unique"]

def test_find_two_unknowns_two_measurements():
    f = inverse.find(_div(V1=12, R1=None, R2=None), [{"kind": "v", "of": "R2", "value": 4}, {"kind": "i", "of": "R1", "value": 0.004}])
    assert f["values"]["R1"] == pytest.approx(2000, rel=1e-5) and f["values"]["R2"] == pytest.approx(1000, rel=1e-5) and f["unique"]

def test_find_unknown_source():
    f = inverse.find(_div(V1=None), [{"kind": "v", "of": "R2", "value": 5}]); assert f["values"]["V1"] == pytest.approx(10, rel=1e-5)

def test_find_by_power_and_node():
    f = inverse.find(_div(R1=None), [{"kind": "p", "of": "R2", "value": 0.00625}]); assert f["values"]["R1"] == pytest.approx(3000, rel=1e-4)
    f = inverse.find(_div(R1=None), [{"kind": "node", "of": 2, "value": 2.5}]); assert f["values"]["R1"] == pytest.approx(3000, rel=1e-5)

def test_underdetermined_is_reported_not_hidden():
    f = inverse.find(_div(R1=None, R2=None), [{"kind": "v", "of": "R2", "value": 5}]); assert not f["unique"] and "NOT unique" in f["message"]

def test_impossible_measurement_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"): inverse.find(_div(R1=None), [{"kind": "v", "of": "R2", "value": 20}])   # more than the supply
    with pytest.raises(ValueError, match="nothing to find"): inverse.find(_div(), [{"kind": "v", "of": "R2", "value": 5}])

def test_ai_can_plan_a_find_but_the_solver_finds():
    def fake(text, as_json=False):
        return {"answer": "", "changes": {}, "constraints": [{"kind": "v", "of": "R2", "value": 2.5}]} if as_json else "R1 is 3 kilohms."
    r = qa.ask(_div(R1=None).to_dict(), "R2 has 2.5 V across it, what is R1?", gen=fake)
    assert r["found"]["values"]["R1"] == pytest.approx(3000, rel=1e-5)

def test_measurement_that_ignores_the_unknown_is_called_out():
    # R1 sits directly across an ideal source: no node voltage can ever reveal it, only a current through R1 can.
    c = Circuit([K("V1", "V", [1, 3], 10), K("R1", "R", [1, 3], None), K("R2", "R", [1, 2], 1000), K("R3", "R", [2, 3], 1000), K("G", "GND", [3])])
    f = inverse.find(c, [{"kind": "node", "of": 2, "value": 5}])
    assert not f["unique"] and "do not depend on R1" in f["message"]
    f = inverse.find(c, [{"kind": "i", "of": "R1", "value": 0.01}])
    assert f["unique"] and f["values"]["R1"] == pytest.approx(1000, rel=1e-5)

# ---- ladder -> PLC wiring --------------------------------------------------------------------
from circuitsense import ladder_wiring
def _tank():
    E = lambda i, t, a, n, y, x=1, coil="normal": {"id": i, "type": t, "addr": a, "nodes": n, "coil": coil, "y": y, "x": x}
    return {"elements": [E("a", "NO", "Man/Auto", [1, 3], 1), E("b", "NO", "Low Sensor Switch", [3, 4], 1, 2), E("c", "NC", "OFF/ON", [4, 5], 1, 3),
                         E("d", "COIL", "Motor", [5, 2], 1, 4), E("e", "NO", "Motor", [1, 6], 2), E("f", "COIL", "G", [6, 2], 2, 2)]}

def test_ladder_to_wiring_roles_and_wires():
    w = ladder_wiring.generate(_tank())
    assert [d["name"] for d in w["inputs"]] == ["Man/Auto", "Low Sensor Switch", "OFF/ON"]          # Motor is driven by a coil -> not an input
    assert [d["name"] for d in w["outputs"]] == ["Motor", "G"]
    tags = {d["name"]: d["tag"] for d in w["inputs"] + w["outputs"]}
    assert tags["Low Sensor Switch"] == "B1" and tags["Man/Auto"] == "S1" and tags["Motor"] == "KM1" and tags["G"] == "H1"
    assert [d["device_contact"] for d in w["inputs"]] == ["NO", "NO", "NC"]
    assert len(w["wires"]) == 2 * 3 + 2 * 2 and len({x["id"] for x in w["wires"]}) == len(w["wires"])
    assert {"from": "24V+", "to": "S1.1"}.items() <= {k: v for k, v in w["wires"][0].items() if k != "id"}.items()
    assert w["svg"].startswith("<svg") and "DI2" in w["svg"] and "KM1" in w["svg"] and w["svg_print"] != w["svg"]
    assert any("Motor" in n and "also" not in n and "internal bit" in n for n in w["notes"])

def test_internal_bits_can_be_excluded_from_io():
    w = ladder_wiring.generate(_tank(), internal=["G"]); assert [d["name"] for d in w["outputs"]] == ["Motor"]

@ocr_ok
def test_wiring_from_parsed_image():
    w = ladder_wiring.generate(analyze_ladder(gen.ladder_image()))
    assert [d["name"] for d in w["inputs"]] == ["X0", "X1"] and [d["name"] for d in w["outputs"]] == ["Y0", "Y1"]

# ---- AC analysis -------------------------------------------------------------------------------
import math, cmath
from circuitsense import acsolve, netlist
def _rc(f_c=1000.0):
    return Circuit([K("V1", "V", [1, 0], 1.0), K("R1", "R", [1, 2], 1000.0), K("C1", "C", [2, 0], 1 / (2 * math.pi * f_c * 1000.0)), K("G", "GND", [0])])

def test_ac_rc_at_cutoff_is_minus3db_minus45deg():
    v = acsolve.ac_solve(_rc(), 1000.0)["node_v"][2]
    assert v["mag"] == pytest.approx(1 / math.sqrt(2), rel=1e-6) and v["phase"] == pytest.approx(-45.0, abs=1e-4)

def test_ac_matches_closed_form_transfer_function():
    for f in (10, 300, 1000, 5000, 1e5):
        h = 1 / complex(1, f / 1000.0); v = acsolve.ac_solve(_rc(), f)["node_v"][2]
        assert v["mag"] == pytest.approx(abs(h), rel=1e-6) and v["phase"] == pytest.approx(math.degrees(cmath.phase(h)), abs=1e-4)

def test_series_rlc_resonance_current_is_v_over_r():
    c = Circuit([K("V1", "V", [1, 0], 10.0), K("R1", "R", [1, 2], 10.0), K("L1", "L", [2, 3], 1e-3), K("C1", "C", [3, 0], 1e-6), K("G", "GND", [0])])
    f0 = 1 / (2 * math.pi * math.sqrt(1e-3 * 1e-6)); r = acsolve.ac_solve(c, f0)
    assert r["parts"]["R1"]["i"]["mag"] == pytest.approx(1.0, rel=1e-6) and r["parts"]["R1"]["i"]["phase"] == pytest.approx(0, abs=1e-3)
    assert r["parts"]["L1"]["q"] == pytest.approx(-r["parts"]["C1"]["q"], rel=1e-6)          # reactive power cancels at resonance
    assert r["parts"]["R1"]["p"] == pytest.approx(5.0, rel=1e-6)                             # 1/2 I^2 R with peak values

def test_ac_power_is_conserved():
    r = acsolve.ac_solve(_rc(), 700.0)["parts"]          # source p = power delivered, R/C p = power absorbed
    assert r["V1"]["p"] == pytest.approx(r["R1"]["p"] + r["C1"]["p"], rel=1e-6) and r["C1"]["p"] == pytest.approx(0, abs=1e-9)
    assert r["V1"]["q"] == pytest.approx(r["C1"]["q"], rel=1e-6)                        # the capacitor supplies all the reactive power

def test_sweep_finds_cutoff_and_resonance():
    s = acsolve.sweep(_rc(), "node:2", 10, 1e5, 400); assert s["f_corner"] == pytest.approx(1000, rel=0.01) and s["f_peak"] is None
    c = Circuit([K("V1", "V", [1, 0], 10.0), K("R1", "R", [1, 2], 10.0), K("L1", "L", [2, 3], 1e-3), K("C1", "C", [3, 0], 1e-6), K("G", "GND", [0])])
    s = acsolve.sweep(c, "v:R1", 500, 50000, 600); assert s["f_peak"] == pytest.approx(5033, rel=0.02)
    assert s["bandwidth"] == pytest.approx(10 / (2 * math.pi * 1e-3), rel=0.03)              # BW = R/(2 pi L)

def test_ac_refuses_what_it_cannot_do():
    with pytest.raises(ValueError, match="non-linear"): acsolve.ac_solve(Circuit([K("V1", "V", [1, 0], 1), K("D1", "D", [1, 0]), K("G", "GND", [0])]), 50)
    with pytest.raises(ValueError, match="above 0"): acsolve.ac_solve(_rc(), 0)

# ---- netlist import (KiCad) ---------------------------------------------------------------------
def test_parse_values():
    P = netlist.parse_value
    assert P("4k7") == 4700 and P("100n") == pytest.approx(1e-7) and P("10Meg") == 1e7 and P("1M", "R") == 1e6 and P("1m") == 1e-3
    assert P("10kΩ", "R") == 1e4 and P("3.3V") == 3.3 and P("2u2") == pytest.approx(2.2e-6) and P("abc") is None

def test_spice_import_solves_ac():
    c = netlist.load(open("samples/rc_lowpass.cir").read()); assert netlist.load(open("samples/rc_lowpass.cir").read()).meta["freq_hint"] == 1000
    assert sorted(x.kind for x in c.components) == ["C", "GND", "R", "V"]
    assert acsolve.ac_solve(c, 1000)["node_v"][2]["mag"] == pytest.approx(1 / math.sqrt(2), rel=1e-3)

def test_kicad_net_import_and_unsupported_parts_are_flagged():
    c = netlist.load(open("samples/divider.net").read())
    assert {x.id: x.kind for x in c.components} == {"Q1": "UNKNOWN", "R1": "R", "R2": "R", "V1": "V", "GND1": "GND"}
    assert [x.value for x in c.components if x.kind == "R"] == [4700.0, 4700.0] and any("Q1" in w for w in c.warnings)
    with pytest.raises(ValueError, match="Unresolved"): solve(c)                              # never solve around a part we could not model
    c.components = [x for x in c.components if x.id != "Q1"]
    assert solve(c)["node_v"][2] == pytest.approx(5.0)                                        # 10 V across 4.7k / 4.7k

def test_kicad_schematic_file_gets_a_helpful_error():
    with pytest.raises(ValueError, match="Export > Netlist"): netlist.load("(kicad_sch (version 2024))")

def test_import_and_ac_endpoints():
    from fastapi.testclient import TestClient
    from circuitsense.api import app
    cl = TestClient(app); r = cl.post("/import", files={"file": ("rc.cir", open("samples/rc_lowpass.cir", "rb"), "text/plain")}); assert r.status_code == 200
    circ = r.json(); assert circ["meta"]["freq_hint"] == 1000
    a = cl.post("/ac", json={"circuit": circ, "freq": 1000}).json(); assert a["node_v"]["2"]["phase"] == pytest.approx(-45, abs=0.1)
    sw = cl.post("/ac/sweep", json={"circuit": circ, "of": "node:2", "f0": 10, "f1": 1e5, "n": 200}).json(); assert sw["f_corner"] == pytest.approx(1000, rel=0.02)
    assert cl.post("/ac", json={"circuit": circ, "freq": -5}).status_code == 422
    assert cl.post("/import", files={"file": ("x.net", b"(kicad_sch)", "text/plain")}).status_code == 422

# ---- transistors and op-amps (hand-calculated textbook answers) --------------------------------------
def Q(i, n, pol="NPN", beta=100): return K(i, "Q", n, params={"polarity": pol, "beta": beta})
def _ce(rb, rc=5000.0, pol="NPN", vcc=10.0, beta=100):
    if pol == "NPN":      # Vcc -> RC -> C ; Vcc -> RB -> B ; E -> gnd
        return Circuit([K("V1", "V", [1, 0], vcc), K("RB", "R", [1, 2], rb), K("RC", "R", [1, 3], rc), Q("Q1", [3, 2, 0], pol, beta), K("G", "GND", [0])])
    return Circuit([K("V1", "V", [1, 0], vcc), K("RB", "R", [2, 0], rb), K("RC", "R", [3, 0], rc), Q("Q1", [3, 2, 1], pol, beta), K("G", "GND", [0])])

def test_bjt_active_region():
    r = solve(_ce(930e3)); q = r["parts"]["Q1"]                        # Ib = 9.3/930k = 10 uA, Ic = 1 mA, Vc = 10 - 5 = 5 V
    assert q["region"] == "active" and q["ib"] == pytest.approx(10e-6, rel=1e-6) and q["i"] == pytest.approx(1e-3, rel=1e-6) and r["node_v"][3] == pytest.approx(5.0, rel=1e-6)

def test_bjt_saturation():
    r = solve(_ce(100e3)); q = r["parts"]["Q1"]                        # beta*Ib = 9.3 mA would need more than Vcc/RC -> saturated
    assert q["region"] == "saturation" and q["v"] == pytest.approx(0.2) and q["i"] == pytest.approx((10 - 0.2) / 5000, rel=1e-6)

def test_bjt_cutoff():
    c = Circuit([K("V1", "V", [1, 0], 10.0), K("VB", "V", [2, 0], 0.4), K("RC", "R", [1, 3], 5000.0), Q("Q1", [3, 2, 0]), K("G", "GND", [0])])     # Vbe = 0.4 V < 0.7 V
    r = solve(c); assert r["parts"]["Q1"]["region"] == "cutoff" and r["node_v"][3] == pytest.approx(10.0) and r["parts"]["Q1"]["i"] == 0

def test_pnp_mirrors_npn():
    r = solve(_ce(930e3, pol="PNP")); q = r["parts"]["Q1"]             # emitter at +10 V: same numbers, mirrored
    assert q["region"] == "active" and r["node_v"][3] == pytest.approx(5.0, rel=1e-6) and q["i"] == pytest.approx(-1e-3, rel=1e-6)   # current INTO the collector terminal is negative for a PNP
    assert solve(_ce(100e3, pol="PNP"))["parts"]["Q1"]["region"] == "saturation"

def _nmos(rd, vg=3.0, vdd=10.0, **p):
    return Circuit([K("VD", "V", [1, 0], vdd), K("VG", "V", [2, 0], vg), K("RD", "R", [1, 3], rd), K("M1", "M", [3, 2, 0], params={"type": "N", "kp": 2e-3, "vth": 1.0, **p}), K("G", "GND", [0])])

def test_nmos_saturation_and_triode():
    r = solve(_nmos(1000)); m = r["parts"]["M1"]                       # Id = 1/2 * 2mA/V^2 * (3-1)^2 = 4 mA, Vds = 6 V
    assert m["region"] == "saturation" and m["i"] == pytest.approx(4e-3, rel=1e-6) and r["node_v"][3] == pytest.approx(6.0, rel=1e-6)
    r = solve(_nmos(5000)); m = r["parts"]["M1"]                       # 5k load pulls it into triode: Id = K[(2)Vds - Vds^2/2], Vds = 10 - 5k Id
    vds = float(min(v for v in np.roots([5, -21, 10]) if 0 < v < 2))                        # Vds = 10 - 10(2Vds - Vds^2/2)  ->  5Vds^2 - 21Vds + 10 = 0, triode root (< Vov = 2)
    assert m["region"] == "triode" and r["node_v"][3] == pytest.approx(vds, rel=1e-6)

def test_nmos_off_and_pmos_mirror():
    assert solve(_nmos(1000, vg=0.5))["parts"]["M1"]["region"] == "cutoff"
    c = Circuit([K("VD", "V", [1, 0], 10.0), K("VG", "V", [2, 0], 7.0), K("RD", "R", [3, 0], 1000.0), K("M1", "M", [3, 2, 1], params={"type": "P", "kp": 2e-3, "vth": 1.0}), K("G", "GND", [0])])
    r = solve(c); assert r["parts"]["M1"]["region"] == "saturation" and r["node_v"][3] == pytest.approx(4.0, rel=1e-6)      # Vsg = 3 V -> 4 mA -> 4 V across RD

def _opamp(rin, rf, vin=1.0, noninv=False, **p):
    if noninv:  # in+ = vin, in- = divider from out
        return Circuit([K("VIN", "V", [1, 0], vin), K("RF", "R", [2, 3], rf), K("RG", "R", [2, 0], rin), K("U1", "OA", [3, 1, 2], params=p), K("G", "GND", [0])])
    return Circuit([K("VIN", "V", [1, 0], vin), K("RIN", "R", [1, 2], rin), K("RF", "R", [2, 3], rf), K("U1", "OA", [3, 0, 2], params=p), K("G", "GND", [0])])

def test_opamp_inverting_and_noninverting_gain():
    assert solve(_opamp(1000, 10000))["node_v"][3] == pytest.approx(-10.0, rel=1e-9)
    assert solve(_opamp(1000, 10000, noninv=True))["node_v"][3] == pytest.approx(11.0, rel=1e-9)

def test_opamp_clips_at_the_rails():
    r = solve(_opamp(1000, 100000, vpos=12.0, vneg=-12.0)); u = r["parts"]["U1"]          # wants -100 V, can only reach -12 V
    assert r["node_v"][3] == pytest.approx(-12.0) and u["region"] == "saturated low"
    assert solve(_opamp(1000, 5000, vpos=12.0, vneg=-12.0))["parts"]["U1"]["region"] == "linear"

def test_vcvs_with_huge_gain_approaches_the_ideal_opamp():
    c = Circuit([K("VIN", "V", [1, 0], 1.0), K("RIN", "R", [1, 2], 1000.0), K("RF", "R", [2, 3], 10000.0), K("E1", "E", [3, 0, 0, 2], 1e7), K("G", "GND", [0])])
    assert solve(c)["node_v"][3] == pytest.approx(-10.0, rel=1e-4)

def test_transistor_circuits_go_through_whatif_and_find():
    w = whatif(_ce(930e3), {"RB": 465e3}); assert w["after"]["parts"]["Q1"]["i"] > w["before"]["parts"]["Q1"]["i"]
    c = _ce(930e3); c.components[1].value = None                                          # unknown base resistor; measured collector voltage 5 V
    f = inverse.find(c, [{"kind": "node", "of": 3, "value": 5.0}]); assert f["values"]["RB"] == pytest.approx(930e3, rel=1e-4)

# ---- AC: ideal op-amp stages work, transistors are refused ---------------------------------------------
def test_ac_active_lowpass_filter():
    c = Circuit([K("VIN", "V", [1, 0], 1.0), K("R1", "R", [1, 2], 1000.0), K("RF", "R", [2, 3], 10000.0), K("CF", "C", [2, 3], 15.9155e-9), K("U1", "OA", [3, 0, 2]), K("G", "GND", [0])])
    s = acsolve.sweep(c, "node:3", 10, 1e5, 400)                                          # gain -10, fc = 1/(2 pi RF CF) = 1 kHz
    assert 10 ** (s["db"][0] / 20) == pytest.approx(10, rel=0.01) and s["f_corner"] == pytest.approx(1000, rel=0.02)

def test_ac_refuses_transistors():
    with pytest.raises(ValueError, match="small-signal"): acsolve.ac_solve(_ce(930e3), 1000)

# ---- netlist import of active parts ----------------------------------------------------------------------
def test_spice_bjt_import_and_solve():
    c = netlist.load(open("samples/ce_amp.cir").read()); q = next(x for x in c.components if x.kind == "Q")
    assert q.nodes == [c.meta["node_names"] and q.nodes[0], q.nodes[1], 0] and q.params == {"polarity": "NPN", "beta": 100.0}
    r = solve(c); assert r["parts"]["Q1"]["region"] == "active" and r["parts"]["Q1"]["i"] == pytest.approx(1e-3, rel=1e-6)

def test_spice_pnp_mosfet_and_vcvs_cards():
    c = netlist.load("""t
V1 vcc 0 10
R1 d 0 1k
M1 d g vcc vcc PM W=10u L=1u
VG g 0 7
.model PM PMOS(KP=20u VTO=-1)
E1 o 0 a b 1e6
.end""")
    m = next(x for x in c.components if x.kind == "M"); assert m.params["type"] == "P" and m.params["kp"] == pytest.approx(200e-6) and m.params["vth"] == 1.0
    assert next(x for x in c.components if x.kind == "E").value == 1e6
    assert netlist.load("t\nQ1 c b e 2N3906\n.end").components[0].params["polarity"] == "PNP"
    assert any("assumed NPN" in w for w in netlist.load("t\nQ1 c b e MYSTERY\n.end").warnings)

def test_kicad_net_opamp_import_and_solve():
    c = netlist.load(open("samples/inverting_opamp.net").read()); assert {x.id: x.kind for x in c.components} == {"R1": "R", "R2": "R", "U1": "OA", "V1": "V", "GND1": "GND"}
    assert solve(c)["node_v"][3] == pytest.approx(-10.0, rel=1e-9) and any("IDEAL op-amp" in w for w in c.warnings)

def test_kicad_net_transistor_pin_roles():
    net = """(export (components (comp (ref "Q1") (value "2N3904") (libsource (lib "Device") (part "Q_NPN_BCE"))))
      (nets (net (code "1") (name "/C") (node (ref "Q1") (pin "2"))) (net (code "2") (name "/B") (node (ref "Q1") (pin "1")))
            (net (code "3") (name "GND") (node (ref "Q1") (pin "3")))))"""
    q = netlist.load(net).components[0]; assert q.kind == "Q" and q.nodes == [1, 2, 0]              # suffix BCE: pin1=B, pin2=C, pin3=E  ->  [C, B, E]

# ---- integration: samples endpoint, wiring pin names, Q&A context, import -> solve through the API ---------
def test_samples_endpoint_lists_and_serves_files():
    from fastapi.testclient import TestClient
    from circuitsense.api import app
    cl = TestClient(app); names = cl.get("/samples").json()
    assert {"ce_amp.cir", "inverting_opamp.net", "divider.net", "rc_lowpass.cir", "ladder.png"} <= set(names)
    assert cl.get("/samples/ce_amp.cir").status_code == 200 and cl.get("/samples/nope.cir").status_code == 404
    assert cl.get("/samples/..%2Frequirements.txt").status_code in (404, 422)          # no path traversal

def test_every_sample_netlist_imports_and_the_dc_ones_solve():
    from fastapi.testclient import TestClient
    from circuitsense.api import app
    cl = TestClient(app)
    for n in ("ce_amp.cir", "inverting_opamp.net", "rc_lowpass.cir"):
        j = cl.post("/import", files={"file": (n, cl.get("/samples/" + n).content)}).json(); assert j["components"], n
        assert cl.post("/solve", json=j).status_code == 200, n
    j = cl.post("/import", files={"file": ("divider.net", cl.get("/samples/divider.net").content)}).json()   # Q1 has a floating collector: flagged, never guessed
    r = cl.post("/solve", json=j); assert r.status_code == 422 and "Unresolved" in r.json()["detail"] and j["warnings"]
    j = cl.post("/import", files={"file": ("ce_amp.cir", cl.get("/samples/ce_amp.cir").content)}).json()
    assert cl.post("/solve", json=j).json()["parts"]["Q1"]["region"] == "active"
    assert cl.post("/ac", json={"circuit": j, "freq": 1000}).status_code == 422       # transistors: clear refusal, not a crash

def test_wiring_names_transistor_and_opamp_pins():
    c = Circuit([K("Q1", "Q", [3, 2, 0], params={"polarity": "NPN"}), K("U1", "OA", [4, 5, 6])])
    nets = wiring.analyze(c)["nets"]; assert "Q1.C" in nets["3"] and "Q1.B" in nets["2"] and "Q1.E" in nets["0"] and "U1.out" in nets["4"]

def test_qa_context_includes_transistor_params_and_region():
    seen = []
    def gen(prompt, as_json=False): seen.append(prompt); return {"answer": "ok", "changes": {}, "constraints": []}
    qa.ask(_ce(930e3).to_dict(), "what region is Q1 in?", gen=gen)
    assert '"polarity": "NPN"' in seen[0] and '"region": "active"' in seen[0]

# ---- deployment safety -------------------------------------------------------------------------------------
def test_ai_endpoints_are_rate_limited(monkeypatch):
    from fastapi.testclient import TestClient
    from circuitsense import api
    monkeypatch.setenv("RATE_LIMIT_AI", "3"); api._hits.clear(); cl = TestClient(api.app)
    codes = [cl.post("/ask", json={"circuit": {"components": []}, "question": "x"}).status_code for _ in range(5)]
    assert 429 not in codes[:3] and codes[3:] == [429, 429]
    assert cl.post("/solve", json={"components": []}).status_code != 429                     # cheap deterministic endpoints are not limited
    assert cl.get("/static/circuitsense-icon.png").status_code == 200 and cl.get("/health").json() == {"ok": True}

def test_env_file_is_loaded_but_never_overrides(tmp_path, monkeypatch):
    from circuitsense import api
    f = tmp_path / ".env"; f.write_bytes("\ufeff# comment\nGEMINI_API_KEY=abc123\nRATE_LIMIT_AI = \"7\"\nEMPTY=\nGEMINI_MODEL=from-file\n".encode("utf-8"))
    for k in ("GEMINI_API_KEY", "RATE_LIMIT_AI", "EMPTY"): monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "from-real-env")
    api._load_env(f); import os
    assert os.environ["GEMINI_API_KEY"] == "abc123" and os.environ["RATE_LIMIT_AI"] == "7" and "EMPTY" not in os.environ and os.environ["GEMINI_MODEL"] == "from-real-env"
    monkeypatch.delenv("GEMINI_API_KEY"); monkeypatch.delenv("RATE_LIMIT_AI")
