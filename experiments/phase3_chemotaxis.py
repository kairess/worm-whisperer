"""S 등급 회전: 가상 유인 물질을 왼쪽(등쪽)/오른쪽에 두고 ASE 입력만으로 진행 방향이 바뀌는지 (AVB 전진 자극 병행)."""
import os, sys, time, numpy as np; sys.path.insert(0, os.getcwd())
from worm.sim import Worm, behavior_descriptors
from worm.env.chem import ChemField, ASEInput
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
for label, src in [("no source", None), ("source left (+y)", (0.0, 3.0)), ("source right (−y)", (0.0, -3.0)), ("source ahead (−x)", (-4.0, 0.0))]:
    t0 = time.time(); w = Worm(NML, "Vfit", motor="boyle"); ase = ASEInput(ChemField(src)) if src else None; cs = []
    for _ in range(int(20 / 0.05)):
        I = {"AVBL": 5.0, "AVBR": 5.0}
        if ase:
            cur, c = ase.currents(np.asarray(w.x)[0], 0.05); I.update(cur); cs.append(c)
        w.step(I)
    d = behavior_descriptors(w.log, skip_s=2.0); xs = np.stack([r["x"] for r in w.log]); c0 = xs[40].mean(0); c1 = xs[-1].mean(0)
    dist = (np.linalg.norm(c0 - np.asarray(src)) - np.linalg.norm(c1 - np.asarray(src))) if src else 0.0
    Imax = max((max(r["I_ext"].get("ASEL", 0), r["I_ext"].get("ASER", 0)) for r in w.log), default=0)
    print(f"{label:18s} v_fwd {d['v_forward']:+.3f} net turn {d['net_turn_deg']:+7.1f}° | 소스 접근 거리 {dist:+.2f} mm | max ASE 전류 {Imax:.1f} pA ({time.time()-t0:.0f}s)")
