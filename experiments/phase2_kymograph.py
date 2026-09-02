"""Phase 2 그림: 곡률 키모그래프 — (a) 진행파 직접 입력(양성 대조군), (b) V1-split + AVB 3 pA, (c) + 고유수용 g20 d2."""
import os, sys, numpy as np; sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from worm.body.rod2d import Rod2D, traveling_wave
from worm.sim import Worm, kinematics_from_log
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
panels = []
rod = Rod2D(); x = rod.initial(); steps = int(0.05 / rod.dt); kap = []; t = 0.0; xs = []
for b in range(200):
    tt = t + np.arange(steps) * rod.dt; x = rod.run(x, traveling_wave(rod, 6.0, 0.9, 0.4, tt)); kap.append(np.asarray(rod.curvature(x))); xs.append(np.asarray(x)); t += 0.05
from worm.body.rod2d import kinematics
k = kinematics(rod, np.stack(xs)[40:], 0.05); panels.append(("direct traveling wave (positive control)", np.array(kap), k["v_axial"]))
for label, kw in [("V1-split, AVB 3 pA", {}), ("V1-split, AVB 3 pA, proprio g20 d2", {"proprio_gain": 20.0, "proprio_offset": 2})]:
    w = Worm(NML, "V1-split", **kw); w.run(10.0, {"AVBL": 3.0, "AVBR": 3.0}); k = kinematics_from_log(w.log)
    panels.append((label, np.stack([r["kappa"] for r in w.log]), k["v_axial"]))
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a, (label, K, v) in zip(ax, panels):
    im = a.imshow(K.T, aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8, extent=[0, K.shape[0] * 0.05, K.shape[1], 0]); a.set_title(f"{label}\nforward speed {v:+.3f} mm/s", fontsize=9); a.set_xlabel("time (s)")
ax[0].set_ylabel("joint (head → tail)"); fig.colorbar(im, ax=ax, label="curvature (1/mm)", shrink=0.8)
os.makedirs("runs/phase2", exist_ok=True); fig.savefig("runs/phase2/fig_kymograph.png", dpi=110); print("saved runs/phase2/fig_kymograph.png")
