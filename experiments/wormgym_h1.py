"""Worm Gym H1: 명령·조향 채널만으로 냄새 소스에 도달하는 정책을 진화 전략(OpenAI-ES)으로 학습한다. docs/PLAN_WORMGYM.md 2절.
사용:
  uv run python experiments/wormgym_h1.py --mode whitelist --channels AVB,AVA,SMDD,SMDV,RIV --gens 60 --out runs/wormgym/h1_cmd
  uv run python experiments/wormgym_h1.py --baseline random|forward --episodes 50           # 하한 대조군
  uv run python experiments/wormgym_h1.py --mode direct --gens 60 --out runs/wormgym/h1_direct  # 상한 대조군 (운동층 직접 조종)
  uv run python experiments/wormgym_h1.py --eval runs/wormgym/h1_cmd/best.npy --episodes 100
"""
import os, sys, json, time, argparse, numpy as np; sys.path.insert(0, os.getcwd())
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# 워커마다 스레드 1개: JAX/XLA CPU 와 BLAS 가 프로세스마다 코어 수만큼 스레드를 띄워 16 워커 × 16 스레드로 초과 예약된다 (부하 평균 100 관측)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "1")
from multiprocessing import Pool
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="whitelist"); ap.add_argument("--channels", default="AVB,AVA,SMDD,SMDV,RIV"); ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--gens", type=int, default=60); ap.add_argument("--pop", type=int, default=32); ap.add_argument("--sigma", type=float, default=0.1); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--ep_per", type=int, default=2); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--episode_s", type=float, default=40.0)
ap.add_argument("--out", default="runs/wormgym/h1"); ap.add_argument("--baseline", default=None); ap.add_argument("--eval", default=None); ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ENV = None; POL = None
def _init(mode, channels, episode_s, hidden):
    global ENV, POL
    from worm.env.gym import WormChemEnv, MLPPolicy
    ENV = WormChemEnv(channels=channels, mode=mode, episode_s=episode_s); POL = MLPPolicy(ENV.n_obs, ENV.n_act, hidden)
def _run(args):
    from worm.env.gym import run_episode
    theta, seed, fixed = args; return run_episode(ENV, POL, theta, seed, fixed)
def summarize(res):
    return {"reach": float(np.mean([r["reached"] for r in res])), "R": float(np.mean([r["R"] for r in res])), "dist": float(np.mean([r["dist"] for r in res])),
            "t_reach": float(np.median([r["t"] for r in res if r["reached"]])) if any(r["reached"] for r in res) else None}
channels = a.channels.split(","); os.makedirs(a.out, exist_ok=True)
from worm.env.gym import WormChemEnv, MLPPolicy
env0 = WormChemEnv(channels=channels, mode=a.mode, episode_s=a.episode_s); pol = MLPPolicy(env0.n_obs, env0.n_act, a.hidden)
pool = Pool(a.workers, initializer=_init, initargs=(a.mode, channels, a.episode_s, a.hidden))
if a.baseline or a.eval:
    if a.baseline == "random": fixed, theta = "random", None
    elif a.baseline == "forward": fixed = [1.0 if c == "AVB" else 0.0 for c in channels] if a.mode == "whitelist" else [1, 0, 0.5, 0]; theta = None
    else: fixed = None; theta = np.load(a.eval)
    t0 = time.time(); res = pool.map(_run, [(theta, 10000 + i, fixed) for i in range(a.episodes)]); s = summarize(res)
    print(f"{a.baseline or a.eval} mode {a.mode} channels {channels} | reach {s['reach']:.2f} R {s['R']:+.2f} dist {s['dist']:.2f} mm t_reach {s['t_reach']} ({time.time()-t0:.0f}s, n={a.episodes})")
    json.dump({"args": vars(a), "summary": s, "episodes": [{k: (float(v) if not isinstance(v, bool) else v) for k, v in r.items()} for r in res]}, open(os.path.join(a.out, f"eval_{a.baseline or 'policy'}.json"), "w"), indent=1)
    sys.exit()
rng = np.random.default_rng(a.seed); theta = rng.normal(0, 0.1, pol.n_params); half = a.pop // 2; log = []; best = (-1e9, theta.copy()); t0 = time.time()
print(f"ES: mode {a.mode} channels {channels} params {pol.n_params} pop {a.pop} ep_per {a.ep_per} episode {a.episode_s}s workers {a.workers}", flush=True)
for g in range(a.gens):
    eps = rng.normal(0, 1, (half, pol.n_params)); eps = np.concatenate([eps, -eps]); cands = theta + a.sigma * eps
    seeds = [1000 * g + i for i in range(a.ep_per)]
    jobs = [(cands[i], s, None) for i in range(a.pop) for s in seeds]; res = pool.map(_run, jobs)
    F = np.array([np.mean([res[i * a.ep_per + j]["R"] for j in range(a.ep_per)]) for i in range(a.pop)])
    reach = np.array([np.mean([res[i * a.ep_per + j]["reached"] for j in range(a.ep_per)]) for i in range(a.pop)])
    ranks = np.argsort(np.argsort(F)); Fn = ranks / (a.pop - 1) - 0.5                      # 순위 정규화
    theta = theta + a.lr / (a.pop * a.sigma) * (eps.T @ Fn) * a.pop                          # OpenAI-ES 갱신
    ib = int(F.argmax())
    if F[ib] > best[0]: best = (float(F[ib]), cands[ib].copy()); np.save(os.path.join(a.out, "best.npy"), best[1])
    np.save(os.path.join(a.out, "theta.npy"), theta)
    rec = {"gen": g, "F_mean": float(F.mean()), "F_max": float(F.max()), "reach_mean": float(reach.mean()), "reach_max": float(reach.max()), "sec": time.time() - t0}; log.append(rec)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print(f"gen {g:3d} F mean {F.mean():+7.2f} max {F.max():+7.2f} | reach mean {reach.mean():.2f} max {reach.max():.2f} | best {best[0]:+.2f} ({time.time()-t0:.0f}s)", flush=True)
