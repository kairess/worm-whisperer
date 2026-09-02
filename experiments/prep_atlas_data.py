"""wormneuroatlas 에서 (1) CeNGEN 수용체 기반 시냅스 극성 예측(Fenyves 2020 방식), (2) Randi 2023 신호전파 아틀라스를
추출해 worm/data/*.npz 로 저장한다. (오프라인 재현성: 이후 단계는 wormneuroatlas 없이 동작)"""
import os, sys, numpy as np
import wormneuroatlas as wna
wna.WormBase.assert_db_version_consistency = lambda self: None
atlas = wna.NeuroAtlas(merge_bilateral=False, merge_dorsoventral=False, merge_numbered=False, merge_AWC=False)
ids = np.array([str(x) for x in atlas.neuron_ids])
nts = wna.SynapseSign().get_neurotransmitters()
out = {"ids": ids, "nts": np.array(nts)}
for mode in ["both", "dominant"]:
    s3 = atlas.get_chemical_synapse_sign(nt_kwargs={"mode": mode})
    out[f"sign3_{mode}"] = s3
    has = ~np.isnan(s3); tot = np.where(has, s3, 0).sum(0)
    out[f"sign_{mode}"] = np.where(has.any(0), np.sign(tot), np.nan)
    print(mode, {v: int((out[f'sign_{mode}'] == v).sum()) for v in [-1, 0, 1]}, "nan", int(np.isnan(out[f"sign_{mode}"]).sum()))
np.savez_compressed("worm/data/synapse_sign.npz", **out)
atlas.load_signal_propagation_atlas()
dff = atlas.get_signal_propagation_map(); q = atlas.get_signal_propagation_q(); occ = atlas.get_signal_propagation_occurrence_matrix()
dff_all = atlas.get_signal_propagation_map_all()
np.savez_compressed("worm/data/randi2023_sigprop.npz", ids=ids, dff=dff, q=q, occ=occ)
meas = np.isfinite(dff).sum(0); print("stimulated neurons with ≥30 measured responders:", int((meas >= 30).sum()), "≥60:", int((meas >= 60).sum()), "any:", int((meas > 0).sum()))
print("top stimulated:", [(ids[j], int(meas[j])) for j in np.argsort(-meas)[:15]])
# 아틀라스 메타데이터 (자극 길이 등)
h5 = atlas.funatlas_h5; print("h5 keys:", list(h5.keys())[:30])
for k in h5.attrs: print("attr", k, h5.attrs[k])
