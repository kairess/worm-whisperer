"""아틀라스 예측 가능성 상한: 시행(trial) 분할 신뢰도. 각 쌍의 시행을 무작위로 두 반으로 나눠 반쪽 평균끼리의
Spearman 상관과, 반쪽 A 로 반쪽 B 의 유의(q) 여부를 예측하는 AUROC 를 계산한다."""
import os, sys, numpy as np, wormneuroatlas as wna
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
wna.WormBase.assert_db_version_consistency = lambda self: None
atlas = wna.NeuroAtlas(merge_bilateral=False, merge_dorsoventral=False, merge_numbered=False, merge_AWC=False)
atlas.load_signal_propagation_atlas(); da = atlas.get_signal_propagation_map_all(); dff = atlas.get_signal_propagation_map(); q = atlas.get_signal_propagation_q()
rng = np.random.default_rng(0); res = {"rho": [], "r": [], "auc": [], "auc_full": []}
for rep in range(50):
    A, B, S, F = [], [], [], []
    for i in range(300):
        for j in range(300):
            x = da[i, j]
            if i == j or x is None or np.size(x) < 2 or not np.isfinite(dff[i, j]): continue
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            if len(x) < 2: continue
            p = rng.permutation(len(x)); h = len(x) // 2
            A.append(x[p[:h]].mean()); B.append(x[p[h:]].mean()); S.append(q[i, j] < 0.05); F.append(dff[i, j])
    A, B, S, F = map(np.array, (A, B, S, F))
    res["rho"].append(spearmanr(A, B)[0]); res["r"].append(pearsonr(A, B)[0])
    res["auc"].append(roc_auc_score(S, np.abs(A))); res["auc_full"].append(roc_auc_score(S, np.abs(F)))
    if rep == 0: print(f"pairs with ≥2 trials: {len(A)}, significant: {int(S.sum())}")
for k, v in res.items(): print(f"{k:9s} mean {np.mean(v):.3f}  [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")
print("해석: rho/r = 반쪽 평균끼리의 상관 (측정 재현성 상한), auc = 반쪽 데이터의 |응답| 로 유의 쌍 분류, auc_full = 전체 평균 |dFF| 로 유의 쌍 분류(유의성이 dFF 로 정의되므로 상한 참고용)")
