"""E10 (수정판): 미로 판정의 시작 자세 흔들림 재평가. 첫 시도(위치 σ 0.1 mm·방향 σ 10°)는 몸이 벽 밖에서 시작하는 자세를 만들었고 냄새가 벽을 통과해 벽 밖 도달이 집계됐다(폐기).
여기서는 worm/env/jitter.valid_poses 로 몸 25 마디가 모두 통로 안에 있는 자세만 쓴다: 통로 축 ±0.3 mm, 옆 σ 0.03 mm, 방향 σ 3°. 에피소드마다 다른 자세(poses) + T-미로는 목표 팔 무작위.
(a) 폭 0.25/0.30/0.35 × L/R 통로 × {배쪽 고정 평지 정책, 우회전 학습 정책} 32 에피소드; (b) T-미로 정책·절제 8 조건 128 에피소드; (c) 격자 최고 개체 64 에피소드.
각 에피소드에서 머리가 열린 셀 밖으로 나간 스텝 비율(head_out)도 기록해 벽 통과를 감시한다. 결과 runs/wormgym/paper/jitter_eval.json"""
import os, sys, json, time, numpy as np; sys.path.insert(0, os.getcwd()); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from worm.env.batch import BatchWormEnv, fit_theta
from worm.env.mazes import MAZES
from worm.env.jitter import valid_poses
JS = "runs/wormgym/paper/jitter_eval.json"; res = json.load(open(JS)) if os.path.exists(JS) else {}
def run(name, theta_path, maze, width, n, episode_s=120.0, **kw):
    if name in res: print(name, "cached", flush=True); return
    mz = MAZES[maze](width=width); env = BatchWormEnv(walls=mz["walls"], starts=mz["starts"], goals=mz["goals"], episode_s=episode_s, **kw)
    th = fit_theta(np.load(theta_path), env); poses = valid_poses(mz, n, seed=hash(name) % 10000); t0 = time.time()
    out = env.rollout(np.stack([th] * n), range(70000, 70000 + n), poses=poses)
    reached = np.asarray(out["reached"]).astype(bool); d = np.asarray(out["d"]); H = np.asarray(out["head"]); cells, cell = mz["cells"], mz["cell"]
    ho = np.array([[ (int(np.floor(H[b, t, 0] / cell + 0.5)), int(np.floor(H[b, t, 1] / cell + 0.5))) not in cells for t in range(H.shape[1])] for b in range(n)]).mean(1)
    tr = np.argmax(d < env.reach_r, 1) * env.dt_action + env.dt_action
    r = {"reach": float(reached.mean()), "n": n, "t_reach_median_s": float(np.median(tr[reached])) if reached.any() else None, "head_out_frac_mean": float(ho.mean()), "head_out_episodes": int((ho > 0.05).sum()), "sec": time.time() - t0}
    if maze == "tmaze":
        first = []
        for b in range(n):
            f = None
            for t in range(H.shape[1]):
                if abs(H[b, t, 0]) > 0.6 and abs(H[b, t, 1]) < 0.3: f = np.sign(H[b, t, 0]); break
            first.append(f)
        dec = [f for f in first if f is not None]; gx = np.asarray(out["src"])[:, 0]
        r.update({"first_entry_decided": len(dec) / n, "first_entry_left": float(np.mean([f < 0 for f in dec])) if dec else None, "goal_left_frac": float((gx < 0).mean()),
                  "reach_goal_left": float(reached[gx < 0].mean()), "reach_goal_right": float(reached[gx > 0].mean()),
                  "t_median_goal_left": float(np.median(tr[(gx < 0) & reached])) if (reached & (gx < 0)).any() else None, "t_median_goal_right": float(np.median(tr[(gx > 0) & reached])) if (reached & (gx > 0)).any() else None})
    res[name] = r; print(name, json.dumps(r), flush=True); json.dump(res, open(JS, "w"), indent=1)
for w in (0.25, 0.30, 0.35):
    for maze in ("corridor", "corridor_R"):
        run(f"w{w}_{maze}_ventral_flat", "runs/wormgym/es_cmd_curr/theta_final.npy", maze, w, 32, touch_obs=True)
        run(f"w{w}_{maze}_rightTrained_ADR017", "runs/wormgym/h5/smddir/corridor_R/theta_final.npy", maze, w, 32, touch_obs=True, omega_smd_dir=True)
T = dict(maze="tmaze", width=0.3, n=128)
run("tmaze_ADR016_odour_touch", "runs/wormgym/h5/tmaze_touch_odor/theta_final.npy", touch_obs=True, **T)
run("tmaze_ADR017", "runs/wormgym/h5/smddir/tmaze/theta_final.npy", touch_obs=True, omega_smd_dir=True, **T)
run("tmaze_lateral_ADR017", "runs/wormgym/h5/lateral/tmaze/theta_final.npy", touch_obs=True, omega_smd_dir=True, lateral_obs=True, **T)
run("tmaze_ADR016_odour_only", "runs/wormgym/h5/tmaze_odor_only/theta_final.npy", touch_obs=False, **T)
run("tmaze_ADR016_touch_only", "runs/wormgym/h5/tmaze_touch_only/theta_final.npy", touch_obs=True, no_odor=True, **T)
run("tmaze_ADR016_odour_touch_zero_touch", "runs/wormgym/h5/tmaze_touch_odor/theta_final.npy", touch_obs=True, zero_touch=True, **T)
run("tmaze_proprio_ADR017", "runs/wormgym/h5/proprio/tmaze/theta_final.npy", touch_obs=True, omega_smd_dir=True, proprio_obs=True, **T)
run("tmaze_flat_ADR016_untrained", "runs/wormgym/es_cmd_curr/theta_final.npy", touch_obs=True, **T)
run("tmaze_lateral_flat_ADR017_untrained", "runs/wormgym/h5/lateral/flat/theta_final.npy", touch_obs=True, omega_smd_dir=True, lateral_obs=True, **T)
run("tmaze_lateral_flat_ADR017_stallgate", "runs/wormgym/h5/lateral/flat/theta_final.npy", touch_obs=True, omega_smd_dir=True, lateral_obs=True, stall_gate=True, **T)
run("grid_best_individual", "runs/wormgym/h5/smddir/grid_fromR/best.npy", "grid", 0.3, 64, episode_s=150.0, touch_obs=True, omega_smd_dir=True)
print("E10_DONE", flush=True)
