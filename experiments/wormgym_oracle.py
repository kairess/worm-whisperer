"""H3 상한: 소스 방향을 아는 오라클 조종기로 운동층을 직접 구동 (커넥톰 우회). 학습된 direct 정책은 '정지' 에 갇혀 상한 역할을 못 했다 (2026-09-03).
조종: 항상 전진 게이트, 머리 편향 = k · sign(소스가 왼쪽이면 +) (소스와의 각도가 작으면 비례), 오메가 없음. 같은 신체·운동층이므로 "이 몸으로 얼마나 빨리 갈 수 있는가" 의 물리적 상한.
사용: uv run python experiments/wormgym_oracle.py --src_dist 2.5 --episodes 256 [--k 1.0]
"""
import os, sys, json, time, argparse, numpy as np; sys.path.insert(0, os.getcwd())
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "1")
from multiprocessing import Pool
ap = argparse.ArgumentParser(); ap.add_argument("--src_dist", type=float, default=2.5); ap.add_argument("--episodes", type=int, default=256); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--k", type=float, default=1.0); ap.add_argument("--omega", action="store_true", help="소스가 뒤(ahead < −0.5)면 오메가턴 발동"); ap.add_argument("--reverse", action="store_true", help="소스가 뒤(ahead < −0.3)면 후진"); ap.add_argument("--episode_s", type=float, default=40.0); ap.add_argument("--out", default="runs/wormgym/oracle"); a = ap.parse_args()
ENV = None
def _init():
    global ENV
    from worm.env.gym import WormChemEnv
    ENV = WormChemEnv(mode="direct", src_dist=a.src_dist, episode_s=a.episode_s)
def _run(seed):
    env = ENV; env.reset(seed); R = 0.0
    while not env.done:
        x = np.asarray(env.w.x); h = x[0] - x[-1]; h /= np.linalg.norm(h) + 1e-12; v = env.field.src - x[0]; v /= np.linalg.norm(v) + 1e-12
        lateral = h[0] * v[1] - h[1] * v[0]; ahead = h[0] * v[0] + h[1] * v[1]
        bias = -a.k * np.clip(lateral * (2.0 if ahead < 0 else 1.0), -1, 1)         # 소스가 왼쪽(lateral>0) → 배쪽 굽힘(κ<0, head_bias<0). 뒤에 있으면 세게
        # direct 행동: [g_f, g_b, head_bias(0..1 → −1..1), omega]
        om = 1.0 if (a.omega and ahead < -0.5) else 0.0; rev = a.reverse and ahead < -0.3
        _, r, done, info = env.step([0.0 if rev else 1.0, 1.0 if rev else 0.0, 0.5 + 0.5 * (-bias if rev else bias), om]); R += r
    return {"R": R, "reached": env.reached, "dist": env.dist(), "t": env.t}
os.makedirs(a.out, exist_ok=True); t0 = time.time()
with Pool(a.workers, initializer=_init) as pool: res = pool.map(_run, [30000 + i for i in range(a.episodes)])
reach = np.mean([r["reached"] for r in res]); tr = np.median([r["t"] for r in res if r["reached"]]) if reach > 0 else None
print(f"oracle direct k {a.k} src {a.src_dist} mm {a.episode_s}s | reach {reach:.3f} t_reach median {tr} final dist {np.mean([r['dist'] for r in res]):.2f} mm R {np.mean([r['R'] for r in res]):+.2f} ({time.time()-t0:.0f}s, n={a.episodes})")
json.dump({"args": vars(a), "reach": float(reach), "t_reach": tr, "episodes": res}, open(os.path.join(a.out, f"oracle_{a.src_dist}mm_k{a.k}{'_omega' if a.omega else ''}{'_rev' if a.reverse else ''}.json"), "w"), indent=1, default=float)
