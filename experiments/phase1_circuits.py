"""Phase 1 회로 실험: 터치 회로와 절제 (Chalfie et al. 1985 와 비교).

각 프로토콜: 지정 뉴런에 정전류(pA) 2000 ms → 마지막 500 ms 의 그룹 평균 막전위를 무자극 대조군과 비교(Δ mV).
사용: uv run python experiments/phase1_circuits.py <net.nml> [--dt 0.05] [--out runs/phase1/circuits_<tag>.md]
"""
import os, re, sys, time, argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network, ablate
from worm.neural import jaxsim

ap = argparse.ArgumentParser()
ap.add_argument("nml"); ap.add_argument("--dt", type=float, default=0.05); ap.add_argument("--dur", type=float, default=2000)
ap.add_argument("--amp", type=float, default=10.0); ap.add_argument("--out", default=None); ap.add_argument("--tag", default="")
ap.add_argument("--variant", default="V0"); ap.add_argument("--theta_class", default=None, help="phase1d log.json: 클래스별 θ63 적용")
a = ap.parse_args()

from worm.neural.variants import make_variant
net0 = load_network(a.nml); net0.pulses = np.zeros((0, 4)); net0, vinfo = make_variant(net0, a.variant); print("variant:", vinfo.get("variant"))
if a.theta_class:
    import json; from worm.neural.variants import apply_theta_class
    net0 = apply_theta_class(net0, json.load(open(a.theta_class))[-1]["theta"]); a.variant += "_" + os.path.basename(os.path.dirname(a.theta_class))
a.tag = f"{a.tag}_{a.variant}"
GROUPS = {"AVA": r"AVA[LR]", "AVB": r"AVB[LR]", "AVD": r"AVD[LR]", "AVE": r"AVE[LR]", "PVC": r"PVC[LR]",
          "A-type(DA,VA)": r"[DV]A\d+", "B-type(DB,VB)": r"[DV]B\d+", "D-type(DD,VD)": r"[DV]D\d+", "AS": r"AS\d+",
          "musc-D": r"MD[LR]\d+", "musc-V": r"MV[LR]\d+"}
gidx = {g: [i for i, n in enumerate(net0.names) if re.fullmatch(p, n)] for g, p in GROUPS.items()}

# (이름, 자극 {뉴런: pA}, 절제 [뉴런])
amp = a.amp
PROTOCOLS = [
    ("control", {}, []),
    ("AVB (forward command)", {"AVBL": amp, "AVBR": amp}, []),
    ("AVA (backward command)", {"AVAL": amp, "AVAR": amp}, []),
    ("PLM (posterior touch)", {"PLML": amp, "PLMR": amp}, []),
    ("ALM+AVM (anterior touch)", {"ALML": amp, "ALMR": amp, "AVM": amp}, []),
    ("ASH (nose touch/aversive)", {"ASHL": amp, "ASHR": amp}, []),
    ("PLM, PVC ablated", {"PLML": amp, "PLMR": amp}, ["PVCL", "PVCR"]),
    ("PLM, AVB ablated", {"PLML": amp, "PLMR": amp}, ["AVBL", "AVBR"]),
    ("ALM+AVM, AVD ablated", {"ALML": amp, "ALMR": amp, "AVM": amp}, ["AVDL", "AVDR"]),
    ("ALM+AVM, AVA ablated", {"ALML": amp, "ALMR": amp, "AVM": amp}, ["AVAL", "AVAR"]),
    ("PLM 5pA", {"PLML": 5.0, "PLMR": 5.0}, []),
    ("PLM 20pA", {"PLML": 20.0, "PLMR": 20.0}, []),
]

def run(stim, abl):
    net = ablate(net0, abl) if abl else net0
    P = jaxsim.build_params(net)
    steps = int(round(a.dur / a.dt)); I = np.zeros((steps, net.n))
    for n, v in stim.items(): I[:, net.index(n)] = v
    _, Vs = jaxsim.simulate(P, a.dt, a.dur, net.n, I_ext=jnp.asarray(I))
    Vs = np.asarray(Vs); last = Vs[int(steps * 0.75):]
    return {g: float(last[:, ii].mean()) for g, ii in gidx.items()}, Vs

rows = []; base = None; t0 = time.time()
for name, stim, abl in PROTOCOLS:
    means, Vs = run(stim, abl)
    if base is None: base = means
    rows.append((name, means))
    print(f"[{time.time()-t0:5.0f}s] {name}")
hdr = "| protocol | " + " | ".join(GROUPS) + " |\n|---|" + "---|" * len(GROUPS) + "\n"
lines = [f"| {rows[0][0]} (abs mV) | " + " | ".join(f"{rows[0][1][g]:.1f}" for g in GROUPS) + " |"]
for name, m in rows[1:]:
    lines.append(f"| {name} | " + " | ".join(f"{m[g]-base[g]:+.1f}" for g in GROUPS) + " |")
md = (f"### 터치 회로 / 절제 실험 ({a.tag or os.path.basename(a.nml)}; dt {a.dt} ms, {a.dur:.0f} ms 자극, 마지막 25% 평균, 기본 {amp} pA)\n\n"
      "값은 대조군 대비 Δ 평균 막전위 (mV). 첫 행은 대조군 절대값.\n\n" + hdr + "\n".join(lines) + "\n")
print(md)
out = a.out or f"runs/phase1/circuits_{a.tag or 'default'}.md"; os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write(md); print("saved", out)
