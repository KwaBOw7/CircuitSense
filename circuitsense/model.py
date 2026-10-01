from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class Component:
    id: str
    kind: str                     # R, C, V, I, L, GND, UNKNOWN
    nodes: list                   # V: [+, -]; GND: [n]; others: [a, b]
    value: Optional[float] = None # ohms / volts / amps
    bbox: Optional[list] = None
    note: str = ""
    phase: float = 0.0            # degrees, AC sources only
    params: dict = field(default_factory=dict)   # Q: polarity NPN|PNP, beta | M: type N|P, kp, vth, lam | OA: vpos, vneg

@dataclass
class Circuit:
    components: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    scale: float = 1.0
    meta: dict = field(default_factory=dict)   # e.g. node names and a frequency hint from an imported netlist
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d):
        return Circuit([Component(**c) for c in d["components"]], d.get("warnings", []), d.get("scale", 1.0), d.get("meta", {}))
