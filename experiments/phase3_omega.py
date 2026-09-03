"""Phase 3 후속: 오메가턴 강화 탐색. RIV 활성 → 앞쪽 omega_rows 행의 배쪽 곡률 편향(k_omega) 을 켜고, 이득·범위·유턴 구간 길이에 따른 순 회전을 잰다.
회전 측정: (a) 머리−꼬리 현(chord) 방향, (b) 마지막 3 s 무게중심 속도 방향 — 둘 다 시작 대비 변화(deg, 반시계 +).
사용: uv run python experiments/phase3_omega.py [--grid quick|full] [--out runs/phase3/omega_scan.txt]
"""
import os, sys, json, time, argparse, itertools, numpy as np; sys.path.insert(0, os.getcwd())
from worm.sim import Worm, behavior_descriptors
from worm.llm.protocols import _stim
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="quick"); ap.add_argument("--out", default="runs/phase3/omega_scan.txt")
ap.add_argument("--k_omega", default=None); ap.add_argument("--rows", default=None); ap.add_argument("--omega_s", default=None); ap.add_argument("--dur", type=float, default=14.0)
ap.add_argument("--T", default=None, help="펄스 전파 시간(s) 목록; 0 = 정적"); ap.add_argument("--sigma", type=float, default=4.0); ap.add_argument("--riv_thresh", type=float, default=10.0)
a = ap.parse_args()
def omega_sched(omega_s, amp=5.0, rev_s=2.0, dur=14.0):
    return [(0, rev_s, _stim(["AVA"], amp)), (rev_s, rev_s + omega_s, {**_stim(["AVB"], amp), **_stim(["SMDV", "RIV"], amp)}), (rev_s + omega_s, dur, _stim(["AVB"], amp))]
def heading_chord(x): h = x[0] - x[-1]; return np.degrees(np.arctan2(h[1], h[0]))
def run(k_omega, rows, omega_s, dur, T=2.0, sigma=4.0):
    w = Worm(NML, "Vfit", motor="boyle"); w.k_omega, w.omega_rows, w.omega_T, w.omega_sigma, w.riv_thresh = k_omega, rows, T, sigma, a.riv_thresh
    w.run_schedule(omega_sched(omega_s, dur=dur), dur); log = w.log
    xs = np.stack([r["x"] for r in log]); c = xs.mean(1)
    h0 = heading_chord(xs[10]); h1 = heading_chord(xs[-1]); d_chord = (h1 - h0 + 180) % 360 - 180
    v1 = c[-1] - c[-60]; d_path = (np.degrees(np.arctan2(v1[1], v1[0])) - h0 + 180) % 360 - 180     # 마지막 3 s 경로 방향 − 초기 머리 방향
    k_ant = np.array([r["kappa"][:12].mean() for r in log]); ob = min(r["readout"].get("omega_bias", 0) for r in log)
    d = behavior_descriptors(log)
    return dict(k_omega=k_omega, rows=rows, omega_s=omega_s, T=T, sigma=sigma, d_chord=float(d_chord), d_path=float(d_path), kappa_ant_min=float(k_ant.min()), omega_bias_min=float(ob),
                v_fwd=d["v_forward"], freq=d["freq"], still=d["frac_still"], disp=float(np.linalg.norm(c[-1] - c[0])))
grids = {"quick": dict(k_omega=[0.0, 0.05, 0.1, 0.2], rows=[12], omega_s=[2.0, 4.0], T=[1.5, 2.5, 4.0]),
         "full": dict(k_omega=[0.0, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3], rows=[12], omega_s=[1.0, 2.0, 3.0, 4.0], T=[1.0, 1.5, 2.5, 4.0])}
g = grids[a.grid]
if a.k_omega: g["k_omega"] = [float(x) for x in a.k_omega.split(",")]
if a.rows: g["rows"] = [int(x) for x in a.rows.split(",")]
if a.omega_s: g["omega_s"] = [float(x) for x in a.omega_s.split(",")]
if a.T: g["T"] = [float(x) for x in a.T.split(",")]
os.makedirs(os.path.dirname(a.out), exist_ok=True); f = open(a.out, "w"); t0 = time.time(); rows = []
for ko, r, os_, T in itertools.product(g["k_omega"], g["rows"], g["omega_s"], g["T"]):
    if ko == 0.0 and (r != g["rows"][0] or T != g["T"][0]): continue
    res = run(ko, r, os_, a.dur, T, a.sigma); rows.append(res)
    line = (f"k_omega {ko:.2f} rows {r:2d} omega {os_:.0f}s T {T:.1f} | Δheading chord {res['d_chord']:+7.1f}° path {res['d_path']:+7.1f}° | κ_ant min {res['kappa_ant_min']:+.2f} bias {res['omega_bias_min']:+.2f} "
            f"| v_fwd {res['v_fwd']:+.3f} f {res['freq']:.2f} still {res['still']:.2f} disp {res['disp']:.2f} mm ({time.time()-t0:.0f}s)")
    print(line, flush=True); f.write(line + "\n"); f.flush()
json.dump(rows, open(a.out.replace(".txt", ".json"), "w"), indent=1)
