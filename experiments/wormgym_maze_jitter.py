"""E10: 미로 판정의 시작 자세 흔들림 재평가. 통로·T-미로 평가는 시작 자세가 고정돼 있어 같은 시드 수만큼 같은 궤적이 반복된다(실질 n=1, T-미로는 목표 2가지).
start_jitter=1.0 (위치 σ 0.1 mm, 방향 σ 10°) 으로 (a) 폭 0.25/0.30/0.35 × L/R 통로 × {배쪽 고정 평지 정책, 우회전 학습 정책} 32 에피소드,
(b) T-미로 첫 진입(|x|>0.6 첫 교차) 128 에피소드 × {ADR-016 냄새+접촉 정책, ADR-017 정책, 좌우 정보 정책} 을 잰다. 결과 runs/wormgym/paper/jitter_eval.json"""
import os, sys, json, time, numpy as np; sys.path.insert(0, os.getcwd()); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from worm.env.batch import BatchWormEnv, fit_theta
from worm.env.mazes import MAZES
JIT = 1.0; res = {}
def run(name, theta_path, maze, width, n, **kw):
    mz = MAZES[maze](width=width); env = BatchWormEnv(walls=mz["walls"], starts=mz["starts"], goals=mz["goals"], touch_obs=True, episode_s=120.0, **kw)
    th = fit_theta(np.load(theta_path), env); t0 = time.time(); out = env.rollout(np.stack([th] * n), range(70000, 70000 + n), start_jitter=JIT)
    reached = np.asarray(out["reached"]).astype(bool); d = np.asarray(out["d"]); H = np.asarray(out["head"])
    r = {"reach": float(reached.mean()), "n": n, "t_reach_median_s": float(np.median(np.argmax(d < env.reach_r, 1)[reached] * env.dt_action + env.dt_action)) if reached.any() else None, "sec": time.time() - t0}
    if maze == "tmaze":
        first = []
        for b in range(n):
            f = None
            for t in range(H.shape[1]):
                if abs(H[b, t, 0]) > 0.6 and abs(H[b, t, 1]) < 0.3: f = np.sign(H[b, t, 0]); break
            first.append(f)
        dec = [f for f in first if f is not None]; r["first_entry_decided"] = len(dec) / n; r["first_entry_left"] = float(np.mean([f < 0 for f in dec])) if dec else None
        gx = np.asarray(out["src"])[:, 0]; r["goal_left_frac"] = float((gx < 0).mean())
        r["reach_goal_left"] = float(reached[gx < 0].mean()); r["reach_goal_right"] = float(reached[gx > 0].mean())
        tr = np.argmax(d < env.reach_r, 1) * env.dt_action + env.dt_action
        r["t_median_goal_left"] = float(np.median(tr[(gx < 0) & reached])) if (reached & (gx < 0)).any() else None; r["t_median_goal_right"] = float(np.median(tr[(gx > 0) & reached])) if (reached & (gx > 0)).any() else None
    res[name] = r; print(name, json.dumps(r), flush=True); json.dump(res, open("runs/wormgym/paper/jitter_eval.json", "w"), indent=1)
for w in (0.25, 0.30, 0.35):
    for maze in ("corridor", "corridor_R"):
        run(f"w{w}_{maze}_ventral_flat", "runs/wormgym/es_cmd_curr/theta_final.npy", maze, w, 32)
        run(f"w{w}_{maze}_rightTrained_ADR017", "runs/wormgym/h5/smddir/corridor_R/theta_final.npy", maze, w, 32, omega_smd_dir=True)
run("tmaze_ADR016_odour_touch", "runs/wormgym/h5/tmaze_touch_odor/theta_final.npy", "tmaze", 0.3, 128)
run("tmaze_ADR017", "runs/wormgym/h5/smddir/tmaze/theta_final.npy", "tmaze", 0.3, 128, omega_smd_dir=True)
run("tmaze_lateral_ADR017", "runs/wormgym/h5/lateral/tmaze/theta_final.npy", "tmaze", 0.3, 128, omega_smd_dir=True, lateral_obs=True)
print("E10_DONE", flush=True)
