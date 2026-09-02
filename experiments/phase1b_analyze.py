"""변형별 신호전파 예측 성능의 통계: 자극 뉴런 단위 부트스트랩 신뢰구간과 변형 간 짝지은 차이.
사용: uv run python experiments/phase1b_analyze.py runs/phase1b/sigprop_V0_*.npz runs/phase1b/sigprop_V1-split_*.npz ...
"""
import sys, os, re, numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
args = sys.argv[1:]; exclude = None
if "--exclude" in args:
    i = args.index("--exclude"); exclude = set(__import__("json").load(open(args[i + 1]))["train"]); del args[i:i + 2]
files = args; rng = np.random.default_rng(0); B = 1000
def load(f):
    d = np.load(f); d = {k: d[k] for k in d.files}
    if exclude:
        ids = [str(x) for x in d["ids"]]; keep = np.array([ids[s] not in exclude for s in d["stim"]])
        d = {k: (v[keep] if k != "ids" else v) for k, v in d.items()}
    return d
def metric_set(dv, dff, sig):
    m = {}
    m["spearman"] = spearmanr(dv, dff)[0]; m["pearson"] = pearsonr(dv, dff)[0]
    m["auroc"] = roc_auc_score(sig, np.abs(dv)) if 0 < sig.sum() < len(sig) else np.nan
    m["pearson_sig"] = pearsonr(dv[sig], dff[sig])[0] if sig.sum() > 2 else np.nan
    # 부호: 유의 쌍 중 실측 억제(dff<0)의 재현율과 흥분(dff>0)의 재현율, 균형 정확도
    neg = sig & (dff < 0); pos = sig & (dff > 0)
    m["inh_recall"] = float((dv[neg] < 0).mean()) if neg.any() else np.nan
    m["exc_recall"] = float((dv[pos] > 0).mean()) if pos.any() else np.nan
    m["sign_bal_acc"] = np.nanmean([m["inh_recall"], m["exc_recall"]])
    return m
def boot(d, pred_key):
    stims = np.unique(d["stim"]); groups = {s: np.where(d["stim"] == s)[0] for s in stims}
    point = metric_set(d[pred_key], d["dff"], d["sig"]); samples = {k: [] for k in point}
    for b in range(B):
        pick = rng.choice(stims, len(stims), replace=True); idx = np.concatenate([groups[s] for s in pick])
        m = metric_set(d[pred_key][idx], d["dff"][idx], d["sig"][idx])
        for k in m: samples[k].append(m[k])
    return point, {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in samples.items()}, samples
rows = []; allsamp = {}
for f in files:
    d = load(f); name = re.sub(r"^sigprop_|\.npz$", "", os.path.basename(f))
    pt, ci, samp = boot(d, "dv"); rows.append((name, pt, ci)); allsamp[name] = samp
    if "anatomy" not in allsamp:
        pt2, ci2, samp2 = boot(d, "anat"); rows.append(("anatomy (direct count)", pt2, ci2)); allsamp["anatomy"] = samp2
d0 = load(files[0]); print(f"significant pairs: {int(d0['sig'].sum())}, of which dff<0: {int((d0['sig'] & (d0['dff']<0)).sum())}")
print(f"{'predictor':38s} " + " ".join(f"{k:>26s}" for k in rows[0][1]))
for name, pt, ci in rows:
    print(f"{name:38s} " + " ".join(f"{pt[k]:+.3f} [{ci[k][0]:+.3f},{ci[k][1]:+.3f}]" for k in pt))
# 짝지은 차이 (같은 부트스트랩 표본 순서를 쓰므로 근사적 짝 비교)
names = [r[0] for r in rows if r[0] != "anatomy (direct count)"]
print("\n짝지은 차이 (변형 − anatomy), 부트스트랩 95% CI:")
for n in names:
    for k in ["spearman", "auroc", "sign_bal_acc"]:
        diff = np.array(allsamp[n][k]) - np.array(allsamp["anatomy"][k])
        print(f"  {n:34s} {k:9s} Δ={np.nanmean(diff):+.3f} [{np.nanpercentile(diff,2.5):+.3f},{np.nanpercentile(diff,97.5):+.3f}]  P(Δ≤0)={np.mean(diff<=0):.3f}")
if len(names) > 1:
    print("\n짝지은 차이 (변형 − 첫 변형):")
    for n in names[1:]:
        for k in ["spearman", "auroc"]:
            diff = np.array(allsamp[n][k]) - np.array(allsamp[names[0]][k])
            print(f"  {n:34s} {k:9s} Δ={np.nanmean(diff):+.3f} [{np.nanpercentile(diff,2.5):+.3f},{np.nanpercentile(diff,97.5):+.3f}]  P(Δ≤0)={np.mean(diff<=0):.3f}")
