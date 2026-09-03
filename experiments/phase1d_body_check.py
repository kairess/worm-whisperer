"""Phase 1d 검증 5: 클래스별 θ63(fit_F)을 Boyle 운동층과 결합했을 때 기존 프로토콜이 유지되는지, 터치 방향이 창발하는지.
Vfit(기준)과 V1-split+θ63 을 같은 프로토콜로 돌려 행동 기술자와 게이트 판독(ΔV_AVB, ΔV_AVA 최대)을 비교한다.
사용: uv run python experiments/phase1d_body_check.py [--theta_class runs/phase1d/fit_F/log.json] [--gate_thresh 3.0] [--out runs/phase1d/body_check_fitF.json]
"""
import os, sys, json, time, argparse, numpy as np; sys.path.insert(0, os.getcwd())
from worm.sim import Worm, behavior_descriptors
from worm.llm.protocols import PROTOCOLS
ap = argparse.ArgumentParser(); ap.add_argument("--theta_class", default="runs/phase1d/fit_F/log.json"); ap.add_argument("--gate_thresh", type=float, default=3.0)
ap.add_argument("--protocols", default="forward,reverse,forward_touch,reverse_touch,turn_left,quiescence_RIS,stop"); ap.add_argument("--dur", type=float, default=12.0)
ap.add_argument("--amp_touch", type=float, default=None, help="터치 프로토콜 진폭 재지정 (기본 5 pA)"); ap.add_argument("--out", default="runs/phase1d/body_check_fitF.json")
a = ap.parse_args()
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
th = json.load(open(a.theta_class))[-1]["theta"]
MODELS = {"Vfit": dict(variant="Vfit"), "fitF": dict(variant="V1-split", theta=th, gate_thresh=a.gate_thresh)}
KEYS = ["v_forward", "frac_backward", "turn_rate", "net_turn_deg", "freq", "frac_still"]
out = {}; t0 = time.time()
for m, kw in MODELS.items():
    for p in a.protocols.split(","):
        fn = PROTOCOLS[p][1]; sch = fn(amp=a.amp_touch) if (a.amp_touch and p.endswith("_touch")) else fn()
        w = Worm(NML, motor="boyle", **kw); w.run_schedule(sch, a.dur); d = behavior_descriptors(w.log)
        ro = [r["readout"] for r in w.log if r.get("readout")]; dB = max(r["dAVB"] for r in ro); dA = max(r["dAVA"] for r in ro)
        rest = float(np.mean(w.V_rest[~w.net.is_muscle()])) if w.V_rest is not None else float("nan")
        out[f"{m}/{p}"] = {**{k: d[k] for k in KEYS}, "max_dAVB": dB, "max_dAVA": dA, "rest_mV": rest}
        print(f"[{time.time()-t0:5.0f}s] {m:5s} {p:15s} v_fwd {d['v_forward']:+.3f} back {d['frac_backward']:.2f} turn {d['turn_rate']:+.1f} still {d['frac_still']:.2f} | max ΔV AVB {dB:+.1f} AVA {dA:+.1f} | rest {rest:.1f}")
json.dump(out, open(a.out, "w"), indent=1); print("saved", a.out)
