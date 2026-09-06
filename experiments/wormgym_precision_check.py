"""E8: 보고 수치의 정밀도 재확인 — 같은 128 시작점을 dt_neural 0.25 ms float32 (탐색·학습 설정) 와 0.05 ms float64 (프로젝트 보고 규칙) 로 평가.
판정(사전 등록, docs/PAPER_OUTLINE.md §7 E8): 미세 스텝 도달률이 거친 스텝 도달률의 Wilson 95 % CI 안에 있고 |Δ| ≤ 0.10 이면 기존 수치 유지."""
import os, sys, json, time, math
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worm.env.batch import BatchWormEnv
from worm.env.mazes import MAZES

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

CASES = {
    "H1_full_2.5mm": dict(theta="runs/wormgym/es_cmd_curr/theta_final.npy", kw=dict(src_dist=2.5), reported=0.914),
    "ADR017_real_2.5mm": dict(theta="runs/wormgym/h5/smddir/flat_far/theta_final.npy", kw=dict(src_dist=2.5, omega_smd_dir=True), reported=0.871),
    "tmaze_left_first": dict(theta="runs/wormgym/h5/smddir/tmaze/theta_final.npy", kw=dict(omega_smd_dir=True, episode_s=120.0, touch_obs=True), maze="tmaze", reported=1.0),
}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 128; only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CASES); seeds = list(range(10000, 10000 + n))
JS = "runs/wormgym/paper/precision_check.json"; res = json.load(open(JS)) if os.path.exists(JS) else {}
for name, c in CASES.items():
    if name not in only: continue
    kw = dict(c["kw"])
    if c.get("maze"):
        mz = MAZES[c["maze"]](width=0.3); kw.update(walls=mz["walls"], starts=mz["starts"], goals=mz["goals"])
    theta = np.load(c["theta"]); res[name] = {"reported": c["reported"]}
    for tag, dt, dtype in (("coarse_0.25ms_f32", 0.25, jnp.float32), ("fine_0.05ms_f64", 0.05, jnp.float64)):
        t0 = time.time(); env = BatchWormEnv(dt_neural=dt, dtype=dtype, **kw)
        out = env.rollout(np.stack([theta] * n).astype(np.float64), seeds); reached = np.asarray(out["reached"]).astype(bool)
        d = np.asarray(out["d"]); t_reach = np.median(np.argmax(d < env.reach_r, 1)[reached] * env.dt_action + env.dt_action) if reached.any() else None
        r = {"reach": float(reached.mean()), "wilson": wilson(int(reached.sum()), n), "t_reach_median_s": None if t_reach is None else float(t_reach),
             "dist_final_mean": float(np.asarray(out["dist"]).mean()), "sec": time.time() - t0}
        if c.get("maze"):
            H = np.asarray(out["head"])
            for thr, key in ((0.6, "first_entry_0.6"), (1.5, "deep_entry_1.5")):      # 0.6: 첫 진입 팔(교차점 폭 0.3 의 두 배 바깥); 1.5: 깊이 들어간 팔(목표 팔과 거의 같음)
                first = []
                for b in range(n):
                    f = None
                    for t in range(H.shape[1]):
                        if abs(H[b, t, 0]) > thr and abs(H[b, t, 1]) < 0.3: f = np.sign(H[b, t, 0]); break
                    first.append(f)
                dec = [f for f in first if f is not None]; r[key + "_decided"] = len(dec) / n; r[key + "_left"] = float(np.mean([f < 0 for f in dec])) if dec else None
            r["goal_left_frac"] = float((np.asarray(out["src"])[:, 0] < 0).mean())
        res[name][tag] = r; print(name, tag, json.dumps(r), flush=True)
    a, b = res[name]["coarse_0.25ms_f32"], res[name]["fine_0.05ms_f64"]; lo, hi = a["wilson"]
    res[name]["verdict"] = "KEEP" if (lo - 1e-9 <= b["reach"] <= hi + 1e-9 and abs(a["reach"] - b["reach"]) <= 0.10) else "REPLACE"   # 1e-9: 도달률 1.0 에서 Wilson 상한 0.999… 부동소수 문제
    if c.get("maze"): res[name]["verdict_left_first"] = "KEEP" if abs(a["first_entry_0.6_left"] - b["first_entry_0.6_left"]) <= 0.10 else "REPLACE"
    json.dump(res, open(JS, "w"), indent=1)
    print(name, "verdict", res[name]["verdict"], f"coarse {a['reach']:.3f} CI [{lo:.3f},{hi:.3f}] fine {b['reach']:.3f}", flush=True)
