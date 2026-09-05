"""H5 미로 정책 분석 (GPU 배치 롤아웃): 도달률·시간, 벽 정면 접촉 시 행동(H5-2: 후진 vs 오메가), 좌/우 접촉 비대칭(H5-4), 재정향 각 분포(H5-5), 궤적 그림.
사용: uv run python experiments/wormgym_maze_analyze.py runs/wormgym/h5/corridor_touch/theta_final.npy --maze corridor [--touch] [--no_odor] [--channels ...] [--episodes 128]
"""
import os, sys, json, argparse, numpy as np; sys.path.insert(0, os.getcwd())
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ap = argparse.ArgumentParser(); ap.add_argument("policy"); ap.add_argument("--maze", required=True); ap.add_argument("--width", type=float, default=0.3); ap.add_argument("--touch", action="store_true")
ap.add_argument("--no_odor", action="store_true"); ap.add_argument("--channels", default="AVB,AVA,SMDD,SMDV,RIV"); ap.add_argument("--episodes", type=int, default=128); ap.add_argument("--episode_s", type=float, default=120.0)
ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--out", default=None); ap.add_argument("--flat", action="store_true", help="같은 정책을 평지(2.5 mm)에서도 평가")
ap.add_argument("--ablate", default=None, help="절제할 채널 (쉼표 구분)"); ap.add_argument("--omega_smd_dir", action="store_true"); ap.add_argument("--zero_touch", action="store_true", help="접촉 관측 절제"); ap.add_argument("--proprio", action="store_true"); ap.add_argument("--start_jitter", type=float, default=0.0); ap.add_argument("--lateral", action="store_true"); ap.add_argument("--stall_gate", action="store_true")
a = ap.parse_args(); ch = a.channels.split(","); ci = {c: i for i, c in enumerate(ch)}; out_dir = a.out or os.path.dirname(a.policy)
from worm.env.batch import BatchWormEnv
from worm.env.mazes import MAZES
mz = MAZES[a.maze](width=a.width)
env = BatchWormEnv(channels=ch, walls=mz["walls"], starts=mz["starts"], goals=mz["goals"], touch_obs=a.touch, no_odor=a.no_odor, episode_s=a.episode_s, hidden=a.hidden, omega_smd_dir=a.omega_smd_dir, zero_touch=a.zero_touch, proprio_obs=a.proprio, lateral_obs=a.lateral, stall_gate=a.stall_gate)
from worm.env.batch import fit_theta
theta = fit_theta(np.load(a.policy), env); mask = None if not a.ablate else np.array([0.0 if c in a.ablate.split(",") else 1.0 for c in ch])
out = env.rollout(np.stack([theta] * a.episodes), range(50000, 50000 + a.episodes), mask, start_jitter=a.start_jitter)
A = out["a"]; O = out["obs"]; D = out["d"]; H = out["head"]; rc = out["reached"]
reach = rc.mean(); t_reach = np.median(np.argmax(D < env.reach_r, 1)[rc] * env.dt_action + env.dt_action) if rc.any() else None
summary = {"maze": a.maze, "reach": float(reach), "t_reach": None if t_reach is None else float(t_reach), "dist": float(out["dist"].mean()), "n": a.episodes}
print(f"{a.policy} | {a.maze} touch={a.touch} odor={not a.no_odor} | reach {reach:.3f} t_reach median {t_reach} final dist {out['dist'].mean():.2f} mm")
# T-미로 결정 지수 (Gourgou 2021 DI): 먼저 도달한 팔 (목표 팔 = 냄새 팔). 목표가 왼쪽/오른쪽으로 무작위이므로 "목표 팔 선택률" 로 보고.
if a.maze == "tmaze":
    arm_reach = []
    for b in range(a.episodes):
        h = H[b]; gx = out["src"][b][0]; first = None
        for t in range(h.shape[0]):
            if abs(h[t, 0]) > 1.5 and abs(h[t, 1]) < 0.3: first = np.sign(h[t, 0]); break
        arm_reach.append(None if first is None else (first == np.sign(gx)))
    dec = [x for x in arm_reach if x is not None]; summary["arm_decided"] = len(dec) / a.episodes; summary["correct_arm"] = float(np.mean(dec)) if dec else None
    print(f"      팔 결정 비율 {len(dec)/a.episodes:.2f}, 목표 팔 선택률 {summary['correct_arm']}  (우연 0.5)")
    # H5-3' (Gourgou 2021 모델링 논문의 "줄기에서부터 일정한 회전 편향"): 줄기 구간(|x|<0.15, y<−0.3)에서 조향 편향 (SMDD−SMDV) 의 부호가 목표 팔 쪽(오른쪽 = 등쪽 = SMDD)과 일치하는가
    if "SMDD" in ci and "SMDV" in ci:
        agree = []; bias_by_side = {1: [], -1: []}
        for b in range(a.episodes):
            h = H[b]; stem = (np.abs(h[:, 0]) < 0.15) & (h[:, 1] < -0.3); gs = np.sign(out["src"][b][0])
            if stem.sum() < 4: continue
            bias = float((A[b][stem, ci["SMDD"]] - A[b][stem, ci["SMDV"]]).mean()); bias_by_side[int(gs)].append(bias); agree.append(np.sign(bias) == gs)
        summary["stem_bias"] = {"agree": float(np.mean(agree)) if agree else None, "mean_bias_goal_right": float(np.mean(bias_by_side[1])) if bias_by_side[1] else None, "mean_bias_goal_left": float(np.mean(bias_by_side[-1])) if bias_by_side[-1] else None}
        print(f"H5-3' 줄기 구간 조향 편향(SMDD−SMDV): 목표 오른쪽일 때 {summary['stem_bias']['mean_bias_goal_right']}, 왼쪽일 때 {summary['stem_bias']['mean_bias_goal_left']}, 부호 일치율 {summary['stem_bias']['agree']}  [판정 > 0.7 이면 '일정한 회전 편향' 기전]")
if a.touch:
    front = O[:, :, -1]; left = O[:, :, -3]; right = O[:, :, -2]
    # H5-2: 벽 정면 접촉(> 0.5) 직후 스텝의 행동
    m = front[:, :-1] > 0.5; nxt = A[:, 1:, :]
    if m.any():
        sel = nxt[m]; p_ava = float((sel[:, ci["AVA"]] * 6 > 3).mean()); p_riv = float((sel[:, ci["RIV"]] * 6 > 3).mean()) if "RIV" in ci else float("nan"); p_avb = float((sel[:, ci["AVB"]] * 6 > 3).mean())
        base = A.reshape(-1, len(ch)); b_ava = float((base[:, ci["AVA"]] * 6 > 3).mean()); b_riv = float((base[:, ci["RIV"]] * 6 > 3).mean()) if "RIV" in ci else float("nan")
        print(f"H5-2  정면 접촉(>0.5) 다음 스텝 (n={m.sum()}): P(AVA>3pA) {p_ava:.2f} (전체 {b_ava:.2f}), P(RIV>3pA) {p_riv:.2f} (전체 {b_riv:.2f}), P(AVB>3pA) {p_avb:.2f}   [판정: AVA ≥ 0.6, RIV < 0.4]")
        summary["h5_2"] = {"n": int(m.sum()), "p_ava": p_ava, "p_riv": p_riv, "p_avb": p_avb, "base_ava": b_ava, "base_riv": b_riv}
    lt, rt = float((left > 0.3).mean()), float((right > 0.3).mean()); summary["touch_lr"] = [lt, rt]
    print(f"H5-4  좌/우 접촉 시간 비율 {lt:.2f} / {rt:.2f} → 비대칭 {max(lt, rt) / (lt + rt + 1e-9):.2f}  [판정 > 0.6]")
# H5-5: 재정향 각 (연속 스텝 머리 이동 방향의 변화) 분포: 벽 근처(접촉 > 0.3) vs 아님
v = np.diff(H, axis=1); ang = np.arctan2(v[..., 1], v[..., 0]); dang = np.abs(np.arctan2(np.sin(np.diff(ang, axis=1)), np.cos(np.diff(ang, axis=1)))); moving = np.linalg.norm(v, axis=2)[:, 1:] > 0.01
if a.touch:
    near = ((O[:, 1:-1, -3:].max(2)) > 0.3)
    big_near = float((dang[near & moving] > np.deg2rad(135)).mean()) if (near & moving).any() else float("nan"); big_far = float((dang[(~near) & moving] > np.deg2rad(135)).mean()) if ((~near) & moving).any() else float("nan")
    print(f"H5-5  큰 재정향(>135°) 비율: 벽 근처 {big_near:.3f} vs 개활 {big_far:.3f}  [eLife 2024: 0.18 → 0.40 에 상응하는 약 2배]"); summary["h5_5"] = {"big_near": big_near, "big_far": big_far}
json.dump(summary, open(os.path.join(out_dir, f"maze_analysis_{a.maze}{'_ablate_' + a.ablate.replace(',', '_') if a.ablate else ''}{'_zerotouch' if a.zero_touch else ''}.json"), "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for w in mz["walls"]: ax.plot([w[0], w[2]], [w[1], w[3]], "k-", lw=0.8)
    for b in range(min(24, a.episodes)): ax.plot(H[b, :, 0], H[b, :, 1], "-", lw=0.8, color="tab:green" if rc[b] else "tab:red", alpha=0.7)
    for g in mz["goals"]: ax.add_patch(plt.Circle(g, env.reach_r, fill=False, color="k"))
    ax.set_aspect("equal"); ax.set_title(f"{a.maze}: reach {reach:.2f}"); fig.tight_layout(); fn = os.path.join(out_dir, f"maze_paths_{a.maze}{'_ablate_' + a.ablate.replace(',', '_') if a.ablate else ''}.png"); fig.savefig(fn, dpi=120); print("saved", fn)
except Exception as e: print("plot skipped:", e)
