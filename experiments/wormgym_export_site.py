"""GitHub Pages 데모용 녹화: 시뮬레이터 롤아웃(50 ms 블록)에서 몸 좌표·뉴런 판독 ΔV·게이트·행동·펄스 부호를 JSON 으로 내보낸다 (docs/assets/scenes/*.json).
페이지는 시뮬레이션을 돌리지 않고 이 녹화를 재생한다."""
import os, sys, json, numpy as np; sys.path.insert(0, os.getcwd()); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from worm.env.batch import BatchWormEnv, fit_theta
from worm.env.mazes import MAZES
CH = ["AVB", "AVA", "SMDD", "SMDV", "RIV"]
SCENES = [
    dict(id="open_field", title="Open-field chemotaxis (lateral-sensing policy)", policy="runs/wormgym/h5/lateral/flat/theta_final.npy", maze=None, seeds=[95003], kw=dict(omega_smd_dir=True, lateral_obs=True), episode_s=40.0,
         note="Whitelisted stimulation of AVB/AVA/SMDD/SMDV/RIV only. Reach 0.988 over 256 random starts, median 16.5 s."),
    dict(id="corridor_L", title="L-corridor (left turn): open-field policy, no maze training", policy="runs/wormgym/es_cmd_curr/theta_final.npy", maze="corridor", seeds=[50000], kw=dict(), episode_s=90.0,
         note="Corner turning = stall → reversal → forward with a deep ventral bend pressed against the wall. Reach 1.00."),
    dict(id="tmaze_right_goal", title="T-maze, food on the RIGHT: enters left first, then corrects by reversing", policy="runs/wormgym/h5/tmaze_odor_only/theta_final.npy", maze="tmaze", seeds=None, goal_side=1, kw=dict(), episode_s=90.0,
         note="With temporal sensing only, the two arms are indistinguishable at the junction; 'go one way, reverse if it gets worse' is the learned strategy. Reach 1.00, but the right goal costs extra time."),
    dict(id="tmaze_left_goal", title="T-maze, food on the LEFT", policy="runs/wormgym/h5/tmaze_odor_only/theta_final.npy", maze="tmaze", seeds=None, goal_side=-1, kw=dict(), episode_s=60.0,
         note="Same policy; the ventral (left) deep bend takes the corner directly."),
    dict(id="corridor_R_dorsal", title="Right-turn corridor: learnable only with dorsal deep bends (ADR-017)", policy="runs/wormgym/h5/smddir/corridor_R/theta_final.npy", maze="corridor_R", seeds=[50000], kw=dict(omega_smd_dir=True), episode_s=90.0,
         note="With ventral-only bends every policy scored 0/32 here. Under the SMDD/SMDV-directed bend rule the agent learns the right corner (reach 1.00)."),
]
def pick_seed(env, th, side):
    for sd in range(80000, 80040):
        rng = np.random.default_rng(sd); rng.uniform(); rng.uniform(); st = env.starts[rng.integers(len(env.starts))]; g = env.goals[rng.integers(len(env.goals))]
        if np.sign(g[0]) == side: return sd
def export(sc):
    kw = dict(sc["kw"]); mz = MAZES[sc["maze"]](width=0.3) if sc["maze"] else None
    env = BatchWormEnv(episode_s=sc["episode_s"], touch_obs=bool(mz), full_trace=True, **({"walls": mz["walls"], "starts": mz["starts"], "goals": mz["goals"]} if mz else {}), **kw)
    th = fit_theta(np.load(sc["policy"]), env); seeds = sc["seeds"] or [pick_seed(env, th, sc["goal_side"])]
    out = env.rollout(np.stack([th] * len(seeds)), seeds); b = 0
    body, dv, gates = out["body"][0][b], out["body"][1][b], out["body"][2][b]                   # (T, nb, 25, 2), (T, nb, 6), (T, nb, 2)
    T, nb = body.shape[:2]; A = out["a"][b]; OS = out["omega_sign"][b]; C = out["c"][b]; D = out["d"][b]
    frames = []
    for t in range(T):
        for k in range(nb):
            frames.append({"t": round(((t * nb + k) + 1) * 0.05, 2), "body": np.round(body[t, k], 3).tolist(), "dv": np.round(dv[t, k], 1).tolist(), "gates": gates[t, k].astype(int).tolist(),
                           "a": np.round(A[t], 2).tolist(), "omega": int(OS[t]), "c": round(float(C[t]), 3), "d": round(float(D[t]), 3)})
        if D[t] < env.reach_r: break
    data = {"id": sc["id"], "title": sc["title"], "note": sc["note"], "channels": CH, "readouts": ["AVB", "AVA", "SMDD", "SMDV", "RIV", "RIS"], "block_s": 0.05, "action_s": 0.5,
            "walls": (mz["walls"].tolist() if mz else []), "src": np.round(out["src"][b], 3).tolist(), "sigma": env.sigma, "reach_r": env.reach_r, "reached": bool(out["reached"][b]), "frames": frames}
    fn = f"docs/assets/scenes/{sc['id']}.json"; json.dump(data, open(fn, "w"), separators=(",", ":")); print(sc["id"], "frames", len(frames), "reached", data["reached"], f"{os.path.getsize(fn)/1e6:.1f} MB", flush=True)
for sc in SCENES: export(sc)
json.dump([{"id": s["id"], "title": s["title"]} for s in SCENES], open("docs/assets/scenes/index.json", "w"))
