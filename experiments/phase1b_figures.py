"""논문용 그림: (a) 모델 ΔV vs 실측 ΔF/F 산점도, (b) 변형별 AUROC/Spearman (부트스트랩 CI), (c) ROC 곡선."""
import os, re, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
args = sys.argv[1:]; exclude = None
if "--exclude" in args:
    i = args.index("--exclude"); exclude = set(__import__("json").load(open(args[i + 1]))["train"]); del args[i:i + 2]
files = args; rng = np.random.default_rng(0)
names = [re.sub(r"^sigprop_|_amp2_stim500(_test)?\.npz$|\.npz$|_V1-split", "", os.path.basename(f)) for f in files]
D = [dict(np.load(f)) for f in files]
if exclude:
    for d in D:
        ids = [str(x) for x in d["ids"]]; keep = np.array([ids[s] not in exclude for s in d["stim"]])
        for k in list(d): 
            if k != "ids": d[k] = d[k][keep]
def boot(x, y, s, B=500):
    stims = np.unique(D[0]["stim"]); groups = {k: np.where(D[0]["stim"] == k)[0] for k in stims}
    au, sp = [], []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.choice(stims, len(stims), replace=True)])
        au.append(roc_auc_score(s[idx], np.abs(x[idx]))); sp.append(spearmanr(x[idx], y[idx])[0])
    return np.percentile(au, [2.5, 97.5]), np.percentile(sp, [2.5, 97.5])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# (a) 산점도: 마지막 파일 (가장 좋은 변형)
d = D[-1]; x, y, s = d["dv"], d["dff"], d["sig"]
ax[0].scatter(x[~s], y[~s], s=3, c="lightgray", label="not significant"); ax[0].scatter(x[s], y[s], s=6, c="tab:red", label="significant (q<0.05)")
ax[0].axhline(0, c="k", lw=.5); ax[0].axvline(0, c="k", lw=.5); ax[0].set_xlabel("model ΔV (mV, 2-s mean)"); ax[0].set_ylabel("measured ⟨ΔF/F⟩ (Randi 2023)")
ax[0].set_title(f"{names[-1]}: Spearman {spearmanr(x, y)[0]:+.3f}"); ax[0].legend(fontsize=8); ax[0].set_xlim(np.percentile(x, [0.5, 99.5])); ax[0].set_ylim(np.percentile(y, [0.2, 99.8]))
# (b) 막대
labels = ["anatomy"] + names; aucs, aucci, sps, spci = [], [], [], []
for lab, dd, key in [("anatomy", D[0], "anat")] + [(n, dd, "dv") for n, dd in zip(names, D)]:
    xx = dd[key]; aucs.append(roc_auc_score(dd["sig"], np.abs(xx))); sps.append(spearmanr(xx, dd["dff"])[0])
    a_ci, s_ci = boot(xx, dd["dff"], dd["sig"]); aucci.append(a_ci); spci.append(s_ci)
pos = np.arange(len(labels)); aucci = np.array(aucci).T; spci = np.array(spci).T
ax[1].bar(pos - 0.2, aucs, 0.4, yerr=[np.array(aucs) - aucci[0], aucci[1] - np.array(aucs)], label="AUROC (significant vs not)", color="tab:blue", capsize=3)
ax1b = ax[1].twinx(); ax1b.bar(pos + 0.2, sps, 0.4, yerr=[np.array(sps) - spci[0], spci[1] - np.array(sps)], label="Spearman ρ", color="tab:orange", capsize=3)
ax[1].axhline(0.802, color="tab:blue", ls=":", lw=1); ax1b.axhline(0.140, color="tab:orange", ls=":", lw=1)
ax[1].text(len(labels) - 0.5, 0.803, "split-half ceiling (AUROC 0.80)", ha="right", va="bottom", fontsize=7, color="tab:blue")
ax1b.text(len(labels) - 0.5, 0.141, "split-half ceiling (ρ 0.14)", ha="right", va="bottom", fontsize=7, color="tab:orange")
ax[1].set_xticks(pos); ax[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8); ax[1].set_ylim(0.5, 0.82); ax1b.set_ylim(0, 0.16)
ax[1].set_ylabel("AUROC"); ax1b.set_ylabel("Spearman ρ"); ax[1].axhline(0.5, c="k", lw=.5, ls="--"); ax[1].set_title("prediction of Randi 2023 signal propagation")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
# (c) ROC
for lab, dd, key in [("anatomy", D[0], "anat")] + [(n, dd, "dv") for n, dd in zip(names, D)]:
    fpr, tpr, _ = roc_curve(dd["sig"], np.abs(dd[key])); ax[2].plot(fpr, tpr, label=f"{lab} ({roc_auc_score(dd['sig'], np.abs(dd[key])):.3f})", lw=1.2)
ax[2].plot([0, 1], [0, 1], "k--", lw=.5); ax[2].set_xlabel("false positive rate"); ax[2].set_ylabel("true positive rate"); ax[2].legend(fontsize=7); ax[2].set_title("ROC: |prediction| → significant pair")
fig.tight_layout(); out = "runs/phase1b/fig_sigprop" + ("_testset" if exclude else "") + ".png"; fig.savefig(out, dpi=110); print("saved", out)
