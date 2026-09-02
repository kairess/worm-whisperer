"""Vfit + Boyle 운동층: 무자극 / AVB / AVA 자극의 곡률 키모그래프와 궤적."""
import os, sys, numpy as np; sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from worm.sim import Worm, kinematics_from_log
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
cases = [("no stimulus", {}), ("AVB 5 pA (forward command)", {"AVBL": 5.0, "AVBR": 5.0}), ("AVA 5 pA (backward command)", {"AVAL": 5.0, "AVAR": 5.0})]
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for j, (label, stim) in enumerate(cases):
    w = Worm(NML, "Vfit", motor="boyle"); w.run(12.0, stim); k = kinematics_from_log(w.log, skip_s=4.0)
    K = np.stack([r["kappa"] for r in w.log]); xs = np.stack([r["x"] for r in w.log])
    im = ax[0, j].imshow(K.T, aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8, extent=[0, K.shape[0] * 0.05, K.shape[1], 0])
    ax[0, j].set_title(f"{label}\nv_axial {k['v_axial']:+.3f} mm/s, {k['freq']:.2f} Hz", fontsize=9); ax[0, j].set_xlabel("time (s)")
    for i in range(0, len(xs), 20): ax[1, j].plot(xs[i, :, 0], xs[i, :, 1], color=plt.cm.viridis(i / len(xs)), lw=1)
    c = xs.mean(1); ax[1, j].plot(c[:, 0], c[:, 1], "k--", lw=0.8, label="centroid"); ax[1, j].set_aspect("equal"); ax[1, j].set_xlabel("x (mm)"); ax[1, j].legend(fontsize=7)
ax[0, 0].set_ylabel("joint (head → tail)"); ax[1, 0].set_ylabel("y (mm)"); fig.colorbar(im, ax=ax[0, :], label="curvature (1/mm)", shrink=0.8)
fig.savefig("runs/phase2/fig_integrated.png", dpi=110); print("saved runs/phase2/fig_integrated.png")
