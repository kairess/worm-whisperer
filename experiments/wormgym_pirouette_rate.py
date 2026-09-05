"""E4(a): 학습 정책의 후진 발작 개시율(/s) 을 dlogC/dt 5분위별로 잰다 — Pierce-Shimomura 1999 (기저 피루엣률 0.025–0.03 /s, dC/dt<0 에서 상승, >0 에서 억제, 시그모이드) 와 비교.
개시 = AVA 행동이 0.5 를 아래→위로 넘는 스텝. 사용: uv run python experiments/wormgym_pirouette_rate.py <policy.npy> [--channels ...] [--omega_smd_dir] [--lateral] [--proprio]"""
import os, sys, json, argparse, numpy as np; sys.path.insert(0, os.getcwd()); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ap = argparse.ArgumentParser(); ap.add_argument("policy"); ap.add_argument("--channels", default="AVB,AVA,SMDD,SMDV,RIV"); ap.add_argument("--omega_smd_dir", action="store_true"); ap.add_argument("--lateral", action="store_true"); ap.add_argument("--proprio", action="store_true")
ap.add_argument("--episodes", type=int, default=128); ap.add_argument("--src_dist", type=float, default=2.5); a = ap.parse_args(); ch = a.channels.split(","); ci = {c: i for i, c in enumerate(ch)}
from worm.env.batch import BatchWormEnv, fit_theta
env = BatchWormEnv(channels=ch, episode_s=40.0, src_dist=a.src_dist, omega_smd_dir=a.omega_smd_dir, lateral_obs=a.lateral, proprio_obs=a.proprio)
th = fit_theta(np.load(a.policy), env); out = env.rollout(np.stack([th] * a.episodes), range(97000, 97000 + a.episodes))
A = out["a"]; C = out["c"]; D = out["d"]; dt = env.dt_action
ava = A[:, :, ci["AVA"]] > 0.5; onset = ava[:, 1:] & (~ava[:, :-1])                                  # (B, T-1)
dc = np.diff(np.log(np.maximum(C, 1e-4)), axis=1) / dt                                                # dlogC/dt 직전 구간 (B, T-1)
active = (D[:, 1:] > env.reach_r)                                                                     # 도달 전 스텝만
q = np.quantile(dc[active], [0.2, 0.4, 0.6, 0.8]); bins = np.digitize(dc, q)
rates = [float(onset[active & (bins == b)].mean() / dt) for b in range(5)]; base = float(onset[active].mean() / dt)
print(f"{a.policy} | reach {out['reached'].mean():.3f} | 후진 개시율 /s by dlogC/dt 분위 1(하강)..5(상승): {np.round(rates, 3).tolist()} | 전체 {base:.3f} /s  [Pierce-Shimomura 1999: 기저 0.025–0.030 /s, 하강에서 ↑ 상승에서 ↓]")
print(f"      하강/상승 비 {rates[0] / max(rates[4], 1e-6):.1f}, 단조성 Spearman ρ = {__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(range(5), rates)[0]:+.2f}")
json.dump({"rates": rates, "baseline": base, "quantiles": q.tolist()}, open(os.path.join(os.path.dirname(a.policy), "pirouette_rate.json"), "w"), indent=1)
