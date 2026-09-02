"""c302가 생성한 NeuroML2 네트워크 파일(.net.nml)을 배열로 파싱한다.

NeuroML이 유일한 진실 원천이다. 커넥톰 리더(SpreadsheetDataReader, Cook2019 등)를 바꿔 c302로
다시 생성하면 여기서 자동 반영된다. 셀 모델 파라미터는 NeuroML의 <cell> 정의와 포함된
cell_C.xml / custom_muscle_components.xml 에서 읽는다.

단위 규약 (NEURON mod 파일과 동일): 전압 mV, 시간 ms, 전류 pA, 전도도 nS, 정전용량 pF, 농도 mM, 면적 cm².
"""
from __future__ import annotations
import math, os, re
from dataclasses import dataclass, field
import numpy as np
import xml.etree.ElementTree as ET

NS = {"n": "http://www.neuroml.org/schema/neuroml2"}

# NeuroML 단위 → 우리 규약. (배율)
_UNITS = {
    "mV": 1.0, "V": 1e3,
    "ms": 1.0, "s": 1e3,
    "pA": 1.0, "nA": 1e3, "uA": 1e6,
    "nS": 1.0, "pS": 1e-3, "uS": 1e3, "mS": 1e6, "S": 1e9,
    "per_ms": 1.0, "per_s": 1e-3, "kHz": 1.0, "Hz": 1e-3,
    "mM": 1.0, "M": 1e3, "uM": 1e-3, "nM": 1e-6,
    "S_per_cm2": 1.0, "mS_per_cm2": 1e-3, "uS_per_cm2": 1e-6,   # 전도도 밀도는 S/cm² 로 유지
    "uF_per_cm2": 1.0,                                            # 비정전용량 µF/cm²
    "um": 1.0,
    "mol_per_m_per_A_per_s": 1.0,
    "": 1.0,
}

def q(s: str) -> float:
    """'0.49 nS' → 0.49 (nS 규약). 무차원은 그대로."""
    s = s.strip()
    m = re.fullmatch(r"([-+0-9.eE]+)\s*([A-Za-z_0-9]*)", s)
    if not m:
        raise ValueError(f"cannot parse quantity {s!r}")
    val, unit = float(m.group(1)), m.group(2)
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r} in {s!r}")
    return val * _UNITS[unit]

def _cell_ref(ref: str) -> str:
    return ref.split("/")[1]   # "../ADAL/0/GenericNeuronCell" → "ADAL"

@dataclass
class GateHH:
    """게이트 x: dx/dt = (x_inf(V) − x)/tau,  x_inf = rate/(1+exp(−(V−mid)/scale)), 기여 = x^instances"""
    name: str; instances: int; tau: float; rate: float; mid: float; scale: float

@dataclass
class CaGate:
    """customHGate: h_inf = 1/(1+exp((ca_half − Ca)/k)), 전도도 계수 = 1 + (h_inf − 1)·alpha (순간)"""
    alpha: float; k: float; ca_half: float

@dataclass
class Channel:
    id: str; gates: list = field(default_factory=list); ca_gate: CaGate | None = None; species: str = ""
    unsupported: str = ""    # 지원하지 않는 게이트 형식이면 사유

@dataclass
class CellType:
    id: str
    area_cm2: float; cap_pF: float; v_init: float
    chans: list            # [(Channel, gmax_S_per_cm2, erev_mV)]
    ca_decay_ms: float; ca_rho: float; ca_rest_mM: float

@dataclass
class Network:
    names: list[str]
    cell_type: np.ndarray            # (N,) int, index into types
    types: list[CellType]
    # 화학 시냅스 (등급형): i_post = w·g·s·(erev − V_post), s_inf = 1/(1+exp((vth − V_pre)/delta)), tau = (1−s_inf)/k
    syn_pre: np.ndarray; syn_post: np.ndarray; syn_w: np.ndarray
    syn_g: np.ndarray; syn_delta: np.ndarray; syn_vth: np.ndarray; syn_k: np.ndarray; syn_erev: np.ndarray
    syn_id: list[str]
    # 갭정션 (대칭): i_a += w·g·(V_b − V_a), i_b += w·g·(V_a − V_b)
    gj_a: np.ndarray; gj_b: np.ndarray; gj_w: np.ndarray; gj_g: np.ndarray
    # 펄스 전류 입력: (cell, delay_ms, dur_ms, amp_pA)
    pulses: np.ndarray
    dt_ms: float | None = None; duration_ms: float | None = None

    @property
    def n(self): return len(self.names)
    def index(self, name): return self.names.index(name)
    def is_muscle(self):
        return np.array([self.types[t].id == "GenericMuscleCell" for t in self.cell_type])

def _parse_channels(doc_roots) -> dict[str, Channel]:
    chans = {}
    for root in doc_roots:
        for ic in root.iter(f"{{{NS['n']}}}ionChannel"):
            ch = Channel(id=ic.get("id"), species=ic.get("species", ""))
            for g in ic.findall("n:gateHHtauInf", NS):
                tc = g.find("n:timeCourse", NS); ss = g.find("n:steadyState", NS)
                if tc is None or ss is None or tc.get("type") != "fixedTimeCourse" or ss.get("type") != "HHSigmoidVariable":
                    ch.unsupported = f"gate {g.get('id')}: {None if tc is None else tc.get('type')}/{None if ss is None else ss.get('type')}"
                    continue
                ch.gates.append(GateHH(g.get("id"), int(g.get("instances")), q(tc.get("tau")),
                                       float(ss.get("rate")), q(ss.get("midpoint")), q(ss.get("scale"))))
            if any(g.tag.split("}")[-1] not in ("gateHHtauInf", "customHGate", "notes") for g in ic):
                ch.unsupported = ch.unsupported or "non-HHtauInf gate"
            cg = ic.find("n:customHGate", NS)
            if cg is not None:
                ch.ca_gate = CaGate(float(cg.get("alpha")), q(cg.get("k")), q(cg.get("ca_half")))
            chans[ch.id] = ch
    return chans

def _parse_cell(cell_el, chans) -> CellType:
    seg = cell_el.find("n:morphology/n:segment", NS)
    p, d = seg.find("n:proximal", NS), seg.find("n:distal", NS)
    diam = float(p.get("diameter"))
    L = math.dist([float(p.get(k)) for k in "xyz"], [float(d.get(k)) for k in "xyz"])
    if L == 0: L = diam                      # 구형 → NEURON은 L=diam 원기둥으로 내보냄
    area_um2 = math.pi * diam * L            # NEURON 원기둥 측면적 (양 끝면 제외)
    area_cm2 = area_um2 * 1e-8
    bp = cell_el.find("n:biophysicalProperties", NS); mp = bp.find("n:membraneProperties", NS)
    cap = q(mp.find("n:specificCapacitance", NS).get("value"))       # µF/cm²
    cap_pF = cap * area_cm2 * 1e6                                      # µF/cm² × cm² = µF → pF
    v_init = q(mp.find("n:initMembPotential", NS).get("value"))
    chan_list = []
    for cd in mp.findall("n:channelDensity", NS):
        ch = chans[cd.get("ionChannel")]
        if ch.unsupported:
            raise NotImplementedError(f"cell {cell_el.get('id')} uses channel {ch.id}: {ch.unsupported}")
        chan_list.append((ch, q(cd.get("condDensity")), q(cd.get("erev"))))
    ip = bp.find("n:intracellularProperties", NS)
    species = ip.find("n:species", NS)
    cm_id = species.get("concentrationModel")
    return CellType(cell_el.get("id"), area_cm2, cap_pF, v_init, chan_list, 0.0, 0.0, 0.0), cm_id

def load_network(nml_path: str) -> Network:
    base = os.path.dirname(os.path.abspath(nml_path))
    root = ET.parse(nml_path).getroot()
    roots = [root] + [ET.parse(os.path.join(base, inc.get("href"))).getroot() for inc in root.findall("n:include", NS)]
    chans = _parse_channels(roots)
    # 농도 모델 (뉴런: fixedFactorConcentrationModel 계열, 근육: muscleConcentrationModel)
    conc = {}
    for r in roots:
        for el in r.iter():
            tag = el.tag.split("}")[-1]
            if tag in ("fixedFactorConcentrationModel", "muscleConcentrationModel", "decayingPoolConcentrationModel"):
                conc[el.get("id")] = (q(el.get("decayConstant")), q(el.get("rho")), q(el.get("restingConc")))
    types, cm_ids = [], []
    for c in root.findall("n:cell", NS):
        ct, cm_id = _parse_cell(c, chans); types.append(ct); cm_ids.append(cm_id)
    for ct, cm_id in zip(types, cm_ids):
        ct.ca_decay_ms, ct.ca_rho, ct.ca_rest_mM = conc[cm_id]
    type_index = {t.id: i for i, t in enumerate(types)}

    syn_defs = {el.get("id"): el for el in root.findall("n:gradedSynapse", NS)}
    gj_defs = {el.get("id"): q(el.get("conductance")) for el in root.findall("n:gapJunction", NS)}
    pulse_defs = {el.get("id"): (q(el.get("delay")), q(el.get("duration")), q(el.get("amplitude")))
                  for el in root.findall("n:pulseGenerator", NS)}

    net = root.find("n:network", NS)
    names, ctype = [], []
    for pop in net.findall("n:population", NS):
        assert pop.get("size") == "1"
        names.append(pop.get("id")); ctype.append(type_index[pop.get("component")])
    idx = {n: i for i, n in enumerate(names)}

    S = {k: [] for k in ["pre", "post", "w", "g", "delta", "vth", "k", "erev", "id"]}
    for proj in net.findall("n:continuousProjection", NS):
        for c in proj.findall("n:continuousConnectionInstanceW", NS):
            sd = syn_defs[c.get("postComponent")]
            S["pre"].append(idx[_cell_ref(c.get("preCell"))]); S["post"].append(idx[_cell_ref(c.get("postCell"))])
            S["w"].append(float(c.get("weight", 1)))
            S["g"].append(q(sd.get("conductance"))); S["delta"].append(q(sd.get("delta")))
            S["vth"].append(q(sd.get("Vth"))); S["k"].append(q(sd.get("k"))); S["erev"].append(q(sd.get("erev")))
            S["id"].append(sd.get("id"))
    G = {k: [] for k in ["a", "b", "w", "g"]}
    for proj in net.findall("n:electricalProjection", NS):
        for c in proj.findall("n:electricalConnectionInstanceW", NS):
            G["a"].append(idx[_cell_ref(c.get("preCell"))]); G["b"].append(idx[_cell_ref(c.get("postCell"))])
            G["w"].append(float(c.get("weight", 1))); G["g"].append(gj_defs[c.get("synapse")])
    pulses = []
    for il in net.findall("n:inputList", NS):
        d, dur, amp = pulse_defs[il.get("component")]
        for inp in il.findall("n:input", NS):
            pulses.append((idx[_cell_ref(inp.get("target"))], d, dur, amp))
    f = lambda k, dt=np.float64: np.asarray(S[k], dtype=dt)
    return Network(
        names=names, cell_type=np.array(ctype), types=types,
        syn_pre=f("pre", np.int32), syn_post=f("post", np.int32), syn_w=f("w"), syn_g=f("g"),
        syn_delta=f("delta"), syn_vth=f("vth"), syn_k=f("k"), syn_erev=f("erev"), syn_id=S["id"],
        gj_a=np.asarray(G["a"], np.int32), gj_b=np.asarray(G["b"], np.int32),
        gj_w=np.asarray(G["w"], np.float64), gj_g=np.asarray(G["g"], np.float64),
        pulses=np.asarray(pulses, np.float64).reshape(-1, 4),
    )

def ablate(net: Network, names: list[str]) -> Network:
    """레이저 절제 모사: 지정 세포의 모든 화학 시냅스(입출력)와 갭정션을 제거한다. 세포 자체는 남는다(전압 기록용)."""
    import copy
    dead = np.array([net.index(n) for n in names])
    keep_s = ~(np.isin(net.syn_pre, dead) | np.isin(net.syn_post, dead))
    keep_g = ~(np.isin(net.gj_a, dead) | np.isin(net.gj_b, dead))
    out = copy.copy(net)
    for k in ["syn_pre", "syn_post", "syn_w", "syn_g", "syn_delta", "syn_vth", "syn_k", "syn_erev"]:
        setattr(out, k, getattr(net, k)[keep_s])
    out.syn_id = [i for i, k in zip(net.syn_id, keep_s) if k]
    for k in ["gj_a", "gj_b", "gj_w", "gj_g"]:
        setattr(out, k, getattr(net, k)[keep_g])
    return out

def lems_settings(lems_path: str):
    """LEMS 파일에서 dt, 길이, 출력 컬럼 순서를 읽는다."""
    root = ET.parse(lems_path).getroot()
    sim = root.find("Simulation")
    cols = {}
    for of in sim.findall("OutputFile"):
        cols[of.get("fileName")] = [c.get("quantity").split("/")[0] for c in of.findall("OutputColumn")]
    return q(sim.get("step")), q(sim.get("length")), cols

if __name__ == "__main__":
    import sys
    net = load_network(sys.argv[1])
    m = net.is_muscle()
    print(f"cells {net.n} (neurons {(~m).sum()}, muscles {m.sum()}), chem {len(net.syn_pre)}, gj {len(net.gj_a)}, pulses {len(net.pulses)}")
    for t in net.types:
        print(f"  {t.id}: area {t.area_cm2*1e8:.1f} um2, C {t.cap_pF:.3f} pF, v0 {t.v_init}, Ca tau {t.ca_decay_ms} rho {t.ca_rho}")
        for ch, gd, e in t.chans:
            print(f"     {ch.id:16s} gmax {gd:.3e} S/cm2 = {gd*t.area_cm2*1e9:.4f} nS, E {e} mV, gates {[(g.name, g.instances, g.tau) for g in ch.gates]}, caGate {ch.ca_gate}")
    from collections import Counter
    print("  syn types:", Counter(net.syn_id))
    pairs = set(zip(net.gj_a.tolist(), net.gj_b.tolist()))
    both = sum((b, a) in pairs for a, b in pairs)
    print(f"  gj directed entries {len(pairs)}, of which reverse also listed: {both}")
    print("  weights chem:", Counter(net.syn_w.tolist()).most_common(5), " gj:", Counter(net.gj_w.tolist()).most_common(5))
