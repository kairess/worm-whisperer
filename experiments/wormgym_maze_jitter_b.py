"""E10b: 나머지 T-미로 판정의 시작 자세 흔들림 재평가 (E10 과 같은 설정): 냄새만 학습 정책, 접촉만 학습 정책, 고유수용 정책, ADR-016 정책의 접촉 관측 절제,
좌우 정보 평지 정책(미로 학습 없음)의 정체 게이트 변형. 결과 runs/wormgym/paper/jitter_eval_b.json"""
import os, sys, json, time, numpy as np; sys.path.insert(0, os.getcwd()); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from worm.env.batch import BatchWormEnv, fit_theta
from worm.env.mazes import MAZES
JIT = 1.0; res = {}; N = 128
def run(name, theta_path, **kw):
    mz = MAZES["tmaze"](width=0.3); env = BatchWormEnv(walls=mz["walls"], starts=mz["starts"], goals=mz["goals"], episode_s=120.0, **kw)
    th = fit_theta(np.load(theta_path), env); t0 = time.time(); out = env.rollout(np.stack([th] * N), range(70000, 70000 + N), start_jitter=JIT)
    reached = np.asarray(out["reached"]).astype(bool); d = np.asarray(out["d"]); H = np.asarray(out["head"]); gx = np.asarray(out["src"])[:, 0]
    first = []
    for b in range(N):
        f = None
        for t in range(H.shape[1]):
            if abs(H[b, t, 0]) > 0.6 and abs(H[b, t, 1]) < 0.3: f = np.sign(H[b, t, 0]); break
        first.append(f)
    dec = [f for f in first if f is not None]; tr = np.argmax(d < env.reach_r, 1) * env.dt_action + env.dt_action
    r = {"reach": float(reached.mean()), "n": N, "first_entry_decided": len(dec) / N, "first_entry_left": float(np.mean([f < 0 for f in dec])) if dec else None,
         "reach_goal_left": float(reached[gx < 0].mean()), "reach_goal_right": float(reached[gx > 0].mean()),
         "t_median_goal_left": float(np.median(tr[(gx < 0) & reached])) if (reached & (gx < 0)).any() else None, "t_median_goal_right": float(np.median(tr[(gx > 0) & reached])) if (reached & (gx > 0)).any() else None, "sec": time.time() - t0}
    res[name] = r; print(name, json.dumps(r), flush=True); json.dump(res, open("runs/wormgym/paper/jitter_eval_b.json", "w"), indent=1)
run("tmaze_ADR016_odour_only", "runs/wormgym/h5/tmaze_odor_only/theta_final.npy", touch_obs=False)
run("tmaze_ADR016_touch_only", "runs/wormgym/h5/tmaze_touch_only/theta_final.npy", touch_obs=True, no_odor=True)
run("tmaze_ADR016_odour_touch_zero_touch", "runs/wormgym/h5/tmaze_touch_odor/theta_final.npy", touch_obs=True, zero_touch=True)
run("tmaze_proprio_ADR017", "runs/wormgym/h5/proprio/tmaze/theta_final.npy", touch_obs=True, omega_smd_dir=True, proprio_obs=True)
run("tmaze_flat_ADR016_untrained", "runs/wormgym/es_cmd_curr/theta_final.npy", touch_obs=True)
run("tmaze_lateral_flat_ADR017_untrained", "runs/wormgym/h5/lateral/flat/theta_final.npy", touch_obs=True, omega_smd_dir=True, lateral_obs=True)
run("tmaze_lateral_flat_ADR017_stallgate", "runs/wormgym/h5/lateral/flat/theta_final.npy", touch_obs=True, omega_smd_dir=True, lateral_obs=True, stall_gate=True)
print("E10B_DONE", flush=True)
