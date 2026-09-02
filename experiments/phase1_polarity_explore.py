"""wormneuroatlas 의 (1) Fenyves 2020 기반 시냅스 극성 예측, (2) Randi 2023 신호전파 아틀라스 구조 탐색."""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
import wormneuroatlas as wna
wna.WormBase.assert_db_version_consistency = lambda self: None   # 오프라인: WormBase 버전 확인 생략
from worm.neural.connectome import load_network
net = load_network("runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
atlas = wna.NeuroAtlas(merge_bilateral=False, merge_dorsoventral=False, merge_numbered=False, merge_AWC=False)
ids = list(atlas.neuron_ids); print("atlas neurons:", len(ids), ids[:8], "...")
missing = [n for n in net.names if not n.startswith("M") and n not in ids]; print("c302 neurons not in atlas:", missing)
nts = wna.SynapseSign().get_neurotransmitters(); print("neurotransmitters:", nts)
sign3 = atlas.get_chemical_synapse_sign()          # sign3[k,i,j] = 전달물질 k 로 j → i 의 예측 부호
for k, nt in enumerate(nts):
    print(f"  {nt:14s} +1 {int((sign3[k]==1).sum()):6d}  −1 {int((sign3[k]==-1).sum()):6d}  0 {int((sign3[k]==0).sum()):6d}  nan {int(np.isnan(sign3[k]).sum()):6d}")
# 층 결합: 예측이 있는 전달물질들의 부호 합 → 부호 (+1/−1), 상충이면 0, 모두 nan 이면 nan
has = ~np.isnan(sign3); tot = np.where(has, sign3, 0).sum(0); sign = np.where(has.any(0), np.sign(tot), np.nan)
print("combined:", {v: int((sign == v).sum()) for v in [-1, 0, 1]}, "nan:", int(np.isnan(sign).sum()))
# 우리 네트워크의 화학 시냅스에 적용
aidx = {n: i for i, n in enumerate(ids)}; aidx["AWCL"] = aidx.get("AWCOFF", -1); aidx["AWCR"] = aidx.get("AWCON", -1)
pre = np.array([aidx.get(net.names[p], -1) for p in net.syn_pre]); post = np.array([aidx.get(net.names[p], -1) for p in net.syn_post])
ok = (pre >= 0) & (post >= 0)
sg = np.full(len(pre), np.nan); sg[ok] = sign[post[ok], pre[ok]]
c302_inh = np.array([("inh" in i) for i in net.syn_id])
nn = ~net.is_muscle()[net.syn_post]
print(f"neuron→neuron chem synapses: {nn.sum()}")
for lab, m in [("c302 exc", nn & ~c302_inh), ("c302 inh", nn & c302_inh)]:
    vals = sg[m]; print(f"  {lab:9s} n={m.sum():5d}: Fenyves +1 {np.sum(vals==1):5d}, −1 {np.sum(vals==-1):5d}, 0(conflict) {np.sum(vals==0):4d}, no-pred {np.isnan(vals).sum():4d}")
# 시냅스 개수 가중 (w) 기준
w = net.syn_w
print(f"  weighted: Fenyves −1 fraction among predicted = {w[nn & (sg==-1)].sum() / w[nn & ~np.isnan(sg) & (sg!=0)].sum():.3f}")
# 신호전파 아틀라스
try:
    atlas.load_signal_propagation_atlas()
    dff = atlas.get_signal_propagation_map(); q = atlas.get_signal_propagation_q(); occ = atlas.get_signal_propagation_occurrence_matrix()
    print("dFF shape", dff.shape, "finite:", int(np.isfinite(dff).sum()), "q<0.05:", int((q < 0.05).sum()), "occurrence>0:", int((occ > 0).sum()))
    # 예: AVA, AVB, PLM, ALM 자극 시 유의한 응답 뉴런
    for s in ["AVAL", "AVBL", "PLML", "ALML", "ASHL", "RIS"]:
        if s in aidx:
            j = aidx[s]; col = dff[:, j]; sig = np.where((q[:, j] < 0.05) & np.isfinite(col))[0]
            top = sorted(sig, key=lambda i: -abs(col[i]))[:8]
            print(f"  stim {s}: measured {int(np.isfinite(col).sum())}, significant {len(sig)} → " + ", ".join(f"{ids[i]}({col[i]:+.2f})" for i in top))
except Exception as e:
    print("signal propagation atlas failed:", repr(e))
