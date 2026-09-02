"""Phase 3: 모든 프로토콜을 실행해 행동 기술자를 표로 만든다."""
import os, sys, json, time, numpy as np; sys.path.insert(0, os.getcwd())
from worm.sim import Worm, behavior_descriptors
from worm.llm.protocols import PROTOCOLS, validate
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
rows = {}
for name, (grade, fn, ref) in PROTOCOLS.items():
    t0 = time.time(); sch = validate(fn()); dur = max([t1 for _, t1, _ in sch] + [10.0])
    w = Worm(NML, "Vfit", motor="boyle"); w.run_schedule(sch, dur); d = behavior_descriptors(w.log)
    ro = w.log[-1]["readout"]; rows[name] = {"grade": grade, **d, "quiescent_frac": float(np.mean([r["readout"].get("quiescent", False) for r in w.log[20:]]))}
    print(f"{name:16s} {grade:4s} v_fwd {d['v_forward']:+.3f} back {d['frac_backward']:.2f} act {d['activity']:.3f} turn {d['turn_rate']:+6.1f}°/s net {d['net_turn_deg']:+6.1f}° kap {d['kappa_amp']:.2f} f {d['freq']:.2f} still {d['frac_still']:.2f} quiet {rows[name]['quiescent_frac']:.2f}  ({time.time()-t0:.0f}s)")
os.makedirs("runs/phase3", exist_ok=True); json.dump(rows, open("runs/phase3/protocol_descriptors.json", "w"), indent=1)
