"""Worm Gym 진화 전략 (OpenAI-ES) — GPU 배치 환경 판. 개체군 전체 × 에피소드를 rollout 한 번으로 평가한다. docs/PLAN_WORMGYM.md.
사용:
  uv run python experiments/wormgym_es.py --mode whitelist --channels AVB,AVA,SMDD,SMDV,RIV --gens 150 --out runs/wormgym/es_cmd
  uv run python experiments/wormgym_es.py --mode direct --gens 150 --out runs/wormgym/es_direct
  uv run python experiments/wormgym_es.py --eval runs/wormgym/es_cmd/best.npy --episodes 256 [--mode ... --channels ...]
  uv run python experiments/wormgym_es.py --baseline random|forward|still --episodes 256
"""
import os, sys, json, time, argparse, numpy as np; sys.path.insert(0, os.getcwd())
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="whitelist"); ap.add_argument("--channels", default="AVB,AVA,SMDD,SMDV,RIV"); ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--gens", type=int, default=150); ap.add_argument("--pop", type=int, default=64); ap.add_argument("--sigma", type=float, default=0.2); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--ep_per", type=int, default=4); ap.add_argument("--episode_s", type=float, default=40.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/wormgym/es"); ap.add_argument("--baseline", default=None); ap.add_argument("--eval", default=None); ap.add_argument("--episodes", type=int, default=256)
ap.add_argument("--living_cost", type=float, default=0.05); ap.add_argument("--terminal_dist_w", type=float, default=2.0)
ap.add_argument("--src_dist", type=float, default=2.5, help="커리큘럼: 가까운 소스(1.5)로 먼저 학습 후 --init 으로 이어서"); ap.add_argument("--init", default=None, help="초기 θ (.npy)")
ap.add_argument("--ablate", default=None, help="H4: 평가 시 절제할 채널 (쉼표 구분), 예 --ablate RIV")
ap.add_argument("--maze", default=None, help="H5: corridor|tmaze|grid"); ap.add_argument("--variant", default="Vfit", help="신경 변형: Vfit | Vfit-shuffle<seed> (E5 대조군)"); ap.add_argument("--width", type=float, default=0.3); ap.add_argument("--touch", action="store_true", help="관측에 벽 접촉 3개 추가")
ap.add_argument("--no_odor", action="store_true", help="H5-3 접촉만 조건: 농도 관측을 0 으로 마스킹"); ap.add_argument("--omega_smd_dir", action="store_true", help="ADR-017 후보: 깊은 굽힘 방향 = SMDD−SMDV 부호"); ap.add_argument("--proprio", action="store_true", help="관측에 머리 곡률(고유수용) 추가"); ap.add_argument("--lateral", action="store_true", help="H5-8: 머리 좌/우 농도차 관측")
ap.add_argument("--dwell", action="store_true", help="H4 정지 과제: 도달해도 계속, 반경 안 스텝마다 +0.5"); ap.add_argument("--move_cost", type=float, default=0.0, help="스텝당 이동 거리(mm) 벌점 계수"); ap.add_argument("--init_channels", default=None, help="--init 의 θ 가 다른 채널 집합이면 그 목록 (새 채널 출력은 0 초기화)")
a = ap.parse_args(); channels = a.channels.split(","); os.makedirs(a.out, exist_ok=True)
from worm.env.batch import BatchWormEnv
maze_kw = {}
if a.maze:
    from worm.env.mazes import MAZES
    mz = MAZES[a.maze](width=a.width); maze_kw = dict(walls=mz["walls"], starts=mz["starts"], goals=mz["goals"], touch_obs=a.touch)
env = BatchWormEnv(variant=a.variant, channels=channels, mode=a.mode, episode_s=a.episode_s, hidden=a.hidden, living_cost=a.living_cost, terminal_dist_w=a.terminal_dist_w, src_dist=a.src_dist, dwell=a.dwell, move_cost=a.move_cost, no_odor=a.no_odor, omega_smd_dir=a.omega_smd_dir, proprio_obs=a.proprio, lateral_obs=a.lateral, **maze_kw)
def pad_theta(th, old_ch, new_ch, nh_old=5, h=16, nh_new=None):
    """작은 채널·관측 집합에서 학습한 θ 를 큰 집합으로 확장: 없던 채널의 출력 가중치 0, 편향 −3 (거의 무자극); 없던 관측(접촉 등)의 입력 가중치 0."""
    nh_new = nh_old if nh_new is None else nh_new
    W1 = th[:nh_old * h].reshape(nh_old, h); b1 = th[nh_old * h:nh_old * h + h]; W2 = th[nh_old * h + h:nh_old * h + h + h * len(old_ch)].reshape(h, len(old_ch)); b2 = th[-len(old_ch):]
    W1n = np.zeros((nh_new, h)); W1n[:nh_old] = W1
    W2n = np.zeros((h, len(new_ch))); b2n = np.full(len(new_ch), -3.0)
    for j, c in enumerate(new_ch):
        if c in old_ch: i = old_ch.index(c); W2n[:, j] = W2[:, i]; b2n[j] = b2[i]
    return np.concatenate([W1n.ravel(), b1, W2n.ravel(), b2n])
def summarize(out):
    rc = out["reached"]; return {"reach": float(rc.mean()), "R": float(out["R"].mean()), "dist": float(out["dist"].mean()), "dwell_frac": float(out["dwell_frac"].mean()),
                                 "t_reach": float(np.median(np.argmax(out["d"] < env.reach_r, 1)[rc] * env.dt_action + env.dt_action)) if rc.any() else None}
if a.baseline or a.eval:
    if a.eval: theta = np.load(a.eval)
    else:
        theta = np.zeros(env.n_params); b = np.full(env.n_act, -30.0)
        if a.baseline == "forward": b[channels.index("AVB") if a.mode == "whitelist" else 0] = 30.0
        theta[-env.n_act:] = b
    B = a.episodes; seeds = list(range(10000, 10000 + B)); t0 = time.time()
    if a.baseline == "random": rng = np.random.default_rng(0); th = rng.normal(0, 3.0, (B, env.n_params))      # 무작위 정책: 큰 무작위 가중치
    else: th = np.stack([theta] * B)
    mask = None
    if a.ablate: mask = np.array([0.0 if c in a.ablate.split(",") else 1.0 for c in channels])
    out = env.rollout(th, seeds, mask); s = summarize(out)
    print(f"{a.baseline or a.eval}{' ablate ' + a.ablate if a.ablate else ''} mode {a.mode} channels {channels} | reach {s['reach']:.3f} R {s['R']:+.2f} dist {s['dist']:.2f} mm t_reach {s['t_reach']} dwell {s['dwell_frac']:.2f} ({time.time()-t0:.0f}s, n={B})")
    json.dump({"args": vars(a), "summary": s}, open(os.path.join(a.out, f"eval_{a.baseline or 'policy'}{'_ablate_' + a.ablate.replace(',', '_') if a.ablate else ''}_{a.src_dist}mm.json"), "w"), indent=1); sys.exit()
rng = np.random.default_rng(a.seed); theta = np.load(a.init) if a.init else rng.normal(0, 0.1, env.n_params); half = a.pop // 2
if a.init and (a.init_channels or theta.shape[0] != env.n_params):
    old_ch = a.init_channels.split(",") if a.init_channels else channels
    nh_old = (theta.shape[0] - a.hidden - a.hidden * len(old_ch) - len(old_ch)) // a.hidden
    theta = pad_theta(theta, old_ch, channels, nh_old, a.hidden, env.n_obs)
assert theta.shape[0] == env.n_params, (theta.shape, env.n_params); log = []; best = (-1e9, theta.copy()); t0 = time.time()
print(f"ES(GPU): mode {a.mode} channels {channels} params {env.n_params} pop {a.pop} ep_per {a.ep_per} episode {a.episode_s}s sigma {a.sigma} lr {a.lr}", flush=True)
for g in range(a.gens):
    eps = rng.normal(0, 1, (half, env.n_params)); eps = np.concatenate([eps, -eps]); cands = theta + a.sigma * eps
    seeds = [1000 * g + j for j in range(a.ep_per)]
    th = np.repeat(cands, a.ep_per, 0); sd = seeds * a.pop; out = env.rollout(th, sd)
    F = out["R"].reshape(a.pop, a.ep_per).mean(1); reach = out["reached"].reshape(a.pop, a.ep_per).mean(1)
    ranks = np.argsort(np.argsort(F)); Fn = ranks / (a.pop - 1) - 0.5
    theta = theta + a.lr / a.sigma * (eps.T @ Fn) / half
    ib = int(F.argmax())
    if F[ib] > best[0]: best = (float(F[ib]), cands[ib].copy()); np.save(os.path.join(a.out, "best.npy"), best[1])
    np.save(os.path.join(a.out, "theta.npy"), theta)
    rec = {"gen": g, "F_mean": float(F.mean()), "F_max": float(F.max()), "reach_mean": float(reach.mean()), "reach_max": float(reach.max()), "sec": time.time() - t0}; log.append(rec)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print(f"gen {g:3d} F mean {F.mean():+7.2f} max {F.max():+7.2f} | reach mean {reach.mean():.2f} max {reach.max():.2f} | best {best[0]:+.2f} ({time.time()-t0:.0f}s)", flush=True)
