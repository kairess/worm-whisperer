"""Phase 4 행동 수준 평가: 검증 문장(프로토콜당 2개) → 번역기 → 스케줄 → 시뮬레이션 → 행동 기술자.
정답 = 같은 프로토콜의 교사 스케줄을 돌린 기술자. 지표: (a) 기술자 공간에서 최근접 프로토콜 일치율, (b) 핵심 축(전진 부호, 회전 부호, 정지) 일치율."""
import os, sys, json, time, numpy as np; sys.path.insert(0, os.getcwd())
from worm.sim import Worm, behavior_descriptors
from worm.llm.phrases import PHRASES
from worm.llm.protocols import PROTOCOLS
from worm.llm import translator as T
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
protos_u = {n: T.schedule_to_u(fn()) for n, (g, fn, r) in PROTOCOLS.items()}; meta = json.load(open("runs/phase4/translator.json"))
tr = T.Translator(protos_u, dim=meta["dim"]).load("runs/phase4/translator.pt"); names = tr.names
KEYS = ["v_forward", "frac_backward", "activity", "turn_rate", "net_turn_deg", "kappa_amp", "freq", "frac_still"]
def run_sched(sch, dur=12.0):
    w = Worm(NML, "Vfit", motor="boyle"); w.run_schedule(sch, dur); return behavior_descriptors(w.log)
ref = json.load(open("runs/phase3/protocol_descriptors.json")) if os.path.exists("runs/phase3/protocol_descriptors.json") else {}
if not ref or "turn_rate" not in next(iter(ref.values())):
    ref = {}
    for n, (g, fn, r) in PROTOCOLS.items(): ref[n] = run_sched(fn()); print("ref", n, {k: round(ref[n][k], 3) for k in ["v_forward", "turn_rate", "frac_still"]})
    json.dump(ref, open("runs/phase3/protocol_descriptors.json", "w"), indent=1)
R = np.array([[ref[n][k] for k in KEYS] for n in names]); mu, sd = R.mean(0), R.std(0) + 1e-6
def nearest(d):
    v = (np.array([d[k] for k in KEYS]) - mu) / sd; Z = (R - mu) / sd; return names[int(np.argmin(np.linalg.norm(Z - v, axis=1)))]
rng = np.random.default_rng(1); rows = []; t0 = time.time()
for n, ph in PHRASES.items():
    for phrase in rng.choice(ph, 2, replace=False):
        u, p, a = tr.predict(T.embed([phrase], meta["embedding"])); pred_proto = names[int(p[0].argmax())]
        d = run_sched(T.u_to_schedule(u[0])); near = nearest(d)
        axis_ok = (np.sign(round(d["v_forward"], 2)) == np.sign(round(ref[n]["v_forward"], 2))) and (abs(d["turn_rate"] - ref[n]["turn_rate"]) < 2.0 or np.sign(d["turn_rate"]) == np.sign(ref[n]["turn_rate"]))
        rows.append({"phrase": phrase, "intended": n, "pred_protocol": pred_proto, "behavior_nearest": near, "axis_ok": bool(axis_ok), "desc": d})
        print(f"[{time.time()-t0:5.0f}s] '{phrase}' ({n}) → proto {pred_proto} | behavior≈{near} | v_fwd {d['v_forward']:+.3f} turn {d['turn_rate']:+.1f} still {d['frac_still']:.2f}")
acc_p = np.mean([r["pred_protocol"] == r["intended"] for r in rows]); acc_b = np.mean([r["behavior_nearest"] == r["intended"] for r in rows]); acc_a = np.mean([r["axis_ok"] for r in rows])
print(f"\n프로토콜 일치 {acc_p:.2f}, 행동 최근접 일치 {acc_b:.2f}, 핵심 축(방향/회전) 일치 {acc_a:.2f}  (n={len(rows)})")
json.dump({"rows": rows, "acc_protocol": acc_p, "acc_behavior": acc_b, "acc_axis": acc_a}, open("runs/phase4/behavior_eval.json", "w"), ensure_ascii=False, indent=1, default=float)
