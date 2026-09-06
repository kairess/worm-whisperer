"""논문 그림 초안 (기존 산출물만 사용). docs/figures/fig2–fig6.png.
fig2 H1 학습 곡선 + 정책 스윕 | fig3 최소 회로 절제 + 피루엣 규칙 | fig4 미로 궤적(녹화 장면) | fig5 굽힘 방향 = 회전 방향 (스왑·규칙·폭) | fig6 좌우 정보와 굽힘 방향 선택 + 배선 대조군"""
import os, sys, json, numpy as np; sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "docs/figures"; os.makedirs(OUT, exist_ok=True); plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def J(p): return json.load(open(p))
# ---- fig2: H1 learning curves + policy sweep
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for name, path, lab in [("near", "runs/wormgym/es_cmd_near/log.json", "stage 1 (1.5 mm)"), ("curr", "runs/wormgym/es_cmd_curr/log.json", "stage 2 (2.5 mm)")]:
    if os.path.exists(path):
        L = J(path); ax[0].plot([r["gen"] + (40 if name == "curr" else 0) for r in L], [r["reach_mean"] for r in L], label=lab)
ax[0].set_xlabel("generation"); ax[0].set_ylabel("reach rate (training seeds)"); ax[0].legend(frameon=False); ax[0].set_title("A  ES learning, command/steering channels")
sw = J("runs/wormgym/es_cmd_curr/analysis.json")["summary"].get("policy_sweep", {})
if sw:
    ds = sorted(float(k) for k in sw); ch = ["AVB", "AVA", "SMDD", "SMDV", "RIV"]
    for c in ch: ax[1].plot(ds, [sw[str(d) if str(d) in sw else k][c] for d, k in zip(ds, sorted(sw, key=float))], marker="o", ms=3, label=c)
    ax[1].axvline(0, color="#999", lw=0.8); ax[1].set_xlabel("10·Δlog C per 0.5 s (falling ← → rising)"); ax[1].set_ylabel("stimulus (fraction of 6 pA)"); ax[1].legend(frameon=False, ncol=2, fontsize=8); ax[1].set_title("B  learned policy: input → stimulation")
pr = J("runs/wormgym/es_cmd_curr/pirouette_rate.json"); ax[2].bar(range(1, 6), pr["rates"], color="#4c72b0"); ax[2].axhline(pr["baseline"], color="#999", ls="--", lw=0.8)
ax[2].set_xlabel("dlogC/dt quintile (1 = falling)"); ax[2].set_ylabel("reversal onset rate (/s)"); ax[2].set_title("C  reversal onsets vs dC/dt (cf. Pierce-Shimomura 1999)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_learning_and_rule.png", dpi=160); plt.close(fig)
# ---- fig3: ablation + minimal circuit
abl = {"none": 0.914, "AVB": 0.434, "AVA": 0.098, "SMDD": 0.410, "SMDV": 0.621, "SMDD+SMDV": 0.273, "RIV": 0.066}
retr = {"full circuit": 0.914, "no RIV (retrained)": 0.191, "no SMD (retrained, best of 3 seeds)": 0.328, "sensory channels only": 0.109, "random policy": 0.04}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].bar(list(abl), list(abl.values()), color=["#555"] + ["#c44e52"] * 6); ax[0].set_ylabel("reach rate (2.5 mm, 40 s, n=256)"); ax[0].set_title("A  channel ablation of the trained policy"); ax[0].tick_params(axis="x", rotation=30)
ax[1].barh(list(retr), list(retr.values()), color=["#4c72b0", "#c44e52", "#dd8452", "#c44e52", "#999"]); ax[1].set_xlabel("reach rate"); ax[1].set_title("B  retraining without a channel"); ax[1].invert_yaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_minimal_circuit.png", dpi=160); plt.close(fig)
# ---- fig4: maze paths from recorded scenes
scenes = [("corridor_L", "L-corridor, open-field policy"), ("tmaze_left_goal", "T-maze, food left"), ("tmaze_right_goal", "T-maze, food right"), ("corridor_R_dorsal", "right corridor, dorsal-bend rule")]
fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
for ax, (sid, title) in zip(axes, scenes):
    d = J(f"docs/assets/scenes/{sid}.json"); F = d["frames"]
    for w in d["walls"]: ax.plot([w[0], w[2]], [w[1], w[3]], color="#444", lw=1)
    H = np.array([f["body"][0] for f in F]); om = np.array([f["omega"] for f in F])
    ax.plot(H[:, 0], H[:, 1], color="#4c72b0", lw=1); ax.scatter(H[om > 0, 0], H[om > 0, 1], s=6, color="#dd8452", label="dorsal bend"); ax.scatter(H[om < 0, 0], H[om < 0, 1], s=6, color="#c44e52", label="ventral bend")
    ax.add_patch(plt.Circle(d["src"], d["reach_r"], fill=False, ls="--", color="#55a868")); ax.set_aspect("equal"); ax.set_title(f"{title}\nreached {d['reached']}, t = {F[-1]['t']:.0f} s", fontsize=9, pad=6); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=7, loc="upper left"); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig4_mazes.png", dpi=160); plt.close(fig)
# ---- fig5: bend direction = turn direction
rows = [("ventral-only bends (ADR-016)", 64, 0, 1.00, 0.00), ("dorsal swap (DOMEGA, control)", 33, 31, None, None), ("SMDD/SMDV-directed bends, right-trained (ADR-017)", None, None, 0.00, 1.00)]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].bar(["ventral-only\nfirst arm L", "ventral-only\nfirst arm R", "dorsal swap\narm L", "dorsal swap\narm R"], [64, 0, 33, 31], color=["#c44e52", "#c44e52", "#dd8452", "#dd8452"]); ax[0].set_ylabel("episodes (n=64)"); ax[0].set_title("A  T-maze arm reached vs bend side")
w = [0.25, 0.30, 0.35]; ax[1].plot(w, [1, 1, 1], "o-", color="#c44e52", label="ventral policy, left corridor"); ax[1].plot(w, [0, 0, 0], "s--", color="#c44e52", label="ventral policy, right corridor")
ax[1].plot(w, [1, 1, 1], "o-", color="#dd8452", label="dorsal-trained policy, right corridor"); ax[1].plot(w, [0, 0, 0], "s--", color="#dd8452", label="dorsal-trained policy, left corridor")
ax[1].set_xlabel("corridor width (mm)"); ax[1].set_ylabel("reach rate"); ax[1].set_ylim(-0.05, 1.1); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("B  handedness is robust to corridor width")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_bend_direction.png", dpi=160); plt.close(fig)
# ---- fig6: lateral info + wiring control
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].bar(["temporal only\n(ADR-017)", "temporal + lateral\ndifference"], [0.61, 0.73], yerr=[0.18, 0.08], capsize=4, color=["#999", "#4c72b0"]); ax[0].axhline(0.5, color="#c44e52", ls="--", lw=0.8); ax[0].set_ylim(0.3, 0.9); ax[0].set_ylabel("P(bend side = source side)"); ax[0].set_title("A  lateral information selects the bend side")   # E9 episode means ± sd (128 episodes)
# E5/E7 (paper/*_far/eval*.txt): retrained / transferred reach; full shuffle = 3 seeds (0.141, 0.145, 0.320 / 0.023, 0.055, 0.047)
labels = ["real\nwiring", "all\nshuffled", "chemical\nonly", "gap junctions\nonly"]; retr = [0.871, np.mean([0.141, 0.145, 0.320]), 0.859, 0.078]; trans = [np.nan, np.mean([0.023, 0.055, 0.047]), 0.914, 0.125]
xx = np.arange(4); w = 0.38
ax[1].bar(xx - w / 2, retr, w, color="#4c72b0", label="retrained"); ax[1].bar(xx + w / 2, trans, w, color="#dd8452", label="real-wiring policy transferred")
ax[1].scatter([1 - w / 2] * 3, [0.141, 0.145, 0.320], color="k", s=10, zorder=3); ax[1].scatter([1 + w / 2] * 3, [0.023, 0.055, 0.047], color="k", s=10, zorder=3)
ax[1].set_xticks(xx); ax[1].set_xticklabels(labels, fontsize=8); ax[1].set_ylim(0, 1.25); ax[1].legend(frameon=False, fontsize=7, loc="upper left", ncol=2); ax[1].set_ylabel("reach rate (2.5 mm, n=256)"); ax[1].set_title("B  wiring shuffles: gap junctions carry the stimulus")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_lateral_and_wiring.png", dpi=160); plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))
