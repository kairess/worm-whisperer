"""Worm Gym 정책 분석: best.npy 를 n 에피소드 평가하고 (a) 도달률·시간·거리, (b) 궤적 그림, (c) H2-1 판정(dC/dt 5분위별 후진/오메가 자극 확률, Spearman ρ),
(d) H2-2 판정(SMDD−SMDV 편향 부호 vs 최근 농도 기울기 부호 일치율)을 낸다. docs/PLAN_WORMGYM.md 2절의 사전 등록 기준.
사용: uv run python experiments/wormgym_analyze.py runs/wormgym/h1_cmd/best.npy --mode whitelist --channels AVB,AVA,SMDD,SMDV,RIV --episodes 100
"""
import os, sys, json, argparse, numpy as np; sys.path.insert(0, os.getcwd())
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "1")
from multiprocessing import Pool
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("policy"); ap.add_argument("--mode", default="whitelist"); ap.add_argument("--channels", default="AVB,AVA,SMDD,SMDV,RIV")
ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--episodes", type=int, default=100); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--episode_s", type=float, default=40.0)
ap.add_argument("--out", default=None); ap.add_argument("--src_dist", type=float, default=2.5); a = ap.parse_args()
channels = a.channels.split(","); out = a.out or os.path.dirname(a.policy); ENV = None; POL = None
def _init():
    global ENV, POL
    from worm.env.gym import WormChemEnv, MLPPolicy
    ENV = WormChemEnv(channels=channels, mode=a.mode, episode_s=a.episode_s, src_dist=a.src_dist); POL = MLPPolicy(ENV.n_obs, ENV.n_act, a.hidden)
def _run(args):
    from worm.env.gym import run_episode
    theta, seed = args; r = run_episode(ENV, POL, theta, seed); r["trace"] = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in t.items()} for t in ENV.trace]; r["src"] = ENV.field.src.tolist()
    nb = int(round(ENV.dt_action * 1000 / ENV.w.block_ms)); L = ENV.w.log                                   # 실현된 게이트: 행동 스텝마다 블록 평균
    r["g_b"] = [float(np.mean([L[i * nb + j]["gates"][1] for j in range(nb)])) for i in range(len(ENV.trace))]
    r["g_f"] = [float(np.mean([L[i * nb + j]["gates"][0] for j in range(nb)])) for i in range(len(ENV.trace))]
    r["omega"] = [float(np.mean([L[i * nb + j]["readout"].get("omega_bias", 0) < 0 for j in range(nb)])) for i in range(len(ENV.trace))]
    return r
theta = np.load(a.policy)
with Pool(a.workers, initializer=_init) as pool: res = pool.map(_run, [(theta, 20000 + i) for i in range(a.episodes)])
reach = np.mean([r["reached"] for r in res]); t_reach = np.median([r["t"] for r in res if r["reached"]]) if reach > 0 else None; dist = np.mean([r["dist"] for r in res])
print(f"policy {a.policy} | reach {reach:.2f} (n={a.episodes}) t_reach median {t_reach} final dist {dist:.2f} mm R {np.mean([r['R'] for r in res]):+.2f}")
# H2-1: dC/dt 5분위별 후진/오메가 자극 확률
rows = []
for r in res:
    tr = r["trace"]; cs = np.array([t["c"] for t in tr]); A = np.array([t["a"] for t in tr]); dc = np.gradient(np.log(np.maximum(cs, 1e-4)), 0.5)
    for i in range(1, len(tr)): rows.append((dc[i - 1], A[i], r["g_b"][i], r["g_f"][i], r["omega"][i]))          # 직전 스텝의 dlogC/dt → 이번 스텝의 행동·게이트
dc = np.array([x[0] for x in rows]); A = np.array([x[1] for x in rows]); GB = np.array([x[2] for x in rows]); GF = np.array([x[3] for x in rows]); OM = np.array([x[4] for x in rows])
q = np.quantile(dc, [0.2, 0.4, 0.6, 0.8]); bins = np.digitize(dc, q)
summary = {"reach": float(reach), "t_reach": None if t_reach is None else float(t_reach), "dist": float(dist), "n": a.episodes}
# H2-1 (사전 등록 지표의 실현판): dlogC/dt 5분위별 실제 후진 게이트 / 오메가 발동 비율. 피루엣 전략이면 1분위(하강)에서 높고 단조 감소.
p_rev = [float(((GB[bins == b] > 0.5) | (OM[bins == b] > 0.5)).mean()) for b in range(5)]; p_b = [float((GB[bins == b] > 0.5).mean()) for b in range(5)]; p_o = [float((OM[bins == b] > 0.5).mean()) for b in range(5)]
p_still = [float(((GB[bins == b] < 0.5) & (GF[bins == b] < 0.5)).mean()) for b in range(5)]
rho = spearmanr(range(5), p_rev)[0]
print("H2-1  실현된 P(후진 게이트 or 오메가 | dlogC/dt 5분위 1(하강)..5(상승)):", np.round(p_rev, 3), f"Spearman ρ = {rho:+.2f}  (판정: ρ < −0.8)")
print("      후진 게이트:", np.round(p_b, 3), " 오메가:", np.round(p_o, 3), " 정지(게이트 없음):", np.round(p_still, 3), " 전체 전진 비율", round(float((GF > 0.5).mean()), 2), "후진", round(float((GB > 0.5).mean()), 2))
summary["h2_1"] = {"p_rev": p_rev, "p_back": p_b, "p_omega": p_o, "p_still": p_still, "rho": float(rho)}
if a.mode == "whitelist":
    ci = {c: i for i, c in enumerate(channels)}
    if "AVA" in ci and "AVB" in ci:
        d_cmd = [float((A[bins == b][:, ci["AVA"]] - A[bins == b][:, ci["AVB"]]).mean()) for b in range(5)]; print("      행동 (AVA−AVB) 평균 by 분위:", np.round(d_cmd, 3), f"ρ = {spearmanr(range(5), d_cmd)[0]:+.2f}")
    if "SMDD" in ci and "SMDV" in ci:
        steer = A[:, ci["SMDD"]] - A[:, ci["SMDV"]]; print(f"      조향 |SMDD−SMDV| > 0.1 비율: {(np.abs(steer) > 0.1).mean():.2f}")
        # H2-2 풍향계: 소스가 머리 기준 왼쪽(+)이면 SMDV(왼쪽 규약) > SMDD 여야 한다. 좌표는 z-성분 부호: heading × (src − head)
        agree = []; n_used = 0
        for r in res:
            tr = r["trace"]; src = np.array(r["src"])
            for i in range(1, len(tr)):
                h = np.array(tr[i - 1]["heading"]); v = src - np.array(tr[i - 1]["head"]); lateral = h[0] * v[1] - h[1] * v[0]     # +: 소스가 왼쪽
                st = tr[i]["a"][ci["SMDV"]] - tr[i]["a"][ci["SMDD"]]                                                              # +: 왼쪽 조향 명령
                if abs(st) > 0.1 and abs(lateral) > 0.2: agree.append(np.sign(st) == np.sign(lateral)); n_used += 1
        h22 = float(np.mean(agree)) if agree else float("nan"); summary["h2_2"] = {"agree": h22, "n": n_used}
        print(f"H2-2  풍향계: 조향 부호 = 소스 쪽 부호 일치율 {h22:.2f} (n={n_used}; 판정 > 0.7, 우연 0.5)")
    print("      채널별 평균 행동:", {c: round(float(A[:, i].mean()), 2) for c, i in ci.items()})
# 정책 함수 스윕 (순환 효과 없이 정책 자체를 읽는다): 변화율 −0.6 … +0.6 (10·Δlog C / 0.5 s), 수준 0.6
if a.mode == "whitelist":
    from worm.env.gym import MLPPolicy
    pol = MLPPolicy(5, len(channels), a.hidden); sweep = {}
    for d in [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]:
        act = pol.act(theta, np.array([0.6, d, 2 * d, 3 * d, 4 * d])); sweep[d] = {c: float(v) for c, v in zip(channels, act)}
        print(f"      정책 스윕 Δlog C/0.5 s = {d:+.1f}: " + " ".join(f"{c} {v:.2f}" for c, v in zip(channels, act)))
    summary["policy_sweep"] = sweep
json.dump({"summary": summary, "episodes": [{k: v for k, v in r.items() if k != "trace"} for r in res]}, open(os.path.join(out, "analysis.json"), "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for r in res[:24]:
        h = np.array([t["head"] for t in r["trace"]]); src = np.array(r["src"]); ax[0].plot(h[:, 0] - src[0], h[:, 1] - src[1], "-", lw=0.8, color="tab:green" if r["reached"] else "tab:red", alpha=0.7)
    ax[0].add_patch(plt.Circle((0, 0), 0.5, fill=False, color="k")); ax[0].set_aspect("equal"); ax[0].set_xlim(-4, 4); ax[0].set_ylim(-4, 4); ax[0].set_title(f"head paths (source at origin), reach {reach:.2f}"); ax[0].set_xlabel("mm")
    ax[1].bar(range(1, 6), summary["h2_1"]["p_rev"] if "h2_1" in summary else [0] * 5); ax[1].set_xlabel("dlogC/dt quintile (1 = falling)"); ax[1].set_ylabel("realized P(reverse gate or omega)"); ax[1].set_title("H2-1")
    fig.tight_layout(); fig.savefig(os.path.join(out, "analysis.png"), dpi=120); print("saved", os.path.join(out, "analysis.png"))
except Exception as e: print("plot skipped:", e)
