"""Phase 0 결과 분석: LEMS OutputFile 컬럼을 파싱해 뉴런/근육 전압을 그린다."""
import os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/phase0"
ref = sys.argv[2] if len(sys.argv) > 2 else "c302_C2_LW_Motor"

def columns(lems, fname):
    root = ET.parse(lems).getroot()
    for of in root.iter("OutputFile"):
        if of.get("fileName") == fname:
            return [c.get("id") for c in of.iter("OutputColumn")]
    raise KeyError(fname)

lems = os.path.join(run_dir, f"LEMS_{ref}.xml")
ncols = columns(lems, f"{ref}.dat")
mfile = os.path.join(run_dir, f"{ref}.muscles.dat"); has_m = os.path.exists(mfile)
mcols = columns(lems, f"{ref}.muscles.dat") if has_m else []
nd = np.loadtxt(os.path.join(run_dir, f"{ref}.dat")); t = nd[:, 0] * 1000; V = nd[:, 1:] * 1000
M = np.loadtxt(mfile)[:, 1:] * 1000 if has_m else np.zeros((len(t), 0))
name = lambda c: c.split("_")[0].replace("v", "")  # e.g. "AVAL_v" -> "AVAL"
ncols = [name(c) for c in ncols]; mcols = [name(c) for c in mcols]
print("neurons:", len(ncols), "muscles:", len(mcols), "t:", t[0], t[-1], "n:", len(t))
idx = {n: i for i, n in enumerate(ncols)}

def group(prefix):
    return [i for n, i in idx.items() if re.fullmatch(prefix + r"\d*", n)]

groups = {"AVB": group("AVB[LR]"), "AVA": group("AVA[LR]"), "PVC": group("PVC[LR]"),
          "DB": group("DB"), "VB": group("VB"), "DA": group("DA"), "VA": group("VA"),
          "DD": group("DD"), "VD": group("VD"), "AS": group("AS")}
# 구간 평균: AVB 자극(0-400ms), 휴지(400-500), AVA 자극(500-1000)
win = {"AVB-on": (50, 400), "rest": (420, 500), "AVA-on": (550, 1000)}
print(f"{'group':6s}" + "".join(f"{w:>12s}" for w in win))
for g, ii in groups.items():
    if not ii: continue
    row = []
    for w, (a, b) in win.items():
        m = (t >= a) & (t < b)
        row.append(V[m][:, ii].mean())
    print(f"{g:6s}" + "".join(f"{v:12.2f}" for v in row))
# 전체 뉴런 요약
print("all-neuron mean mV per window:", [round(float(V[(t>=a)&(t<b)].mean()),2) for a,b in win.values()])
for n in ncols[:6]:
    i = idx[n]; print(f"  {n:6s} min {V[:,i].min():7.2f} max {V[:,i].max():7.2f} end {V[-1,i]:7.2f}")
if not has_m:
    fig, ax = plt.subplots(figsize=(10,4)); [ax.plot(t, V[:, idx[n]], label=n) for n in ncols[:9]]; ax.legend(); ax.set_xlabel("ms"); ax.set_ylabel("mV")
    out = os.path.join(run_dir, f"{ref}.png"); fig.savefig(out, dpi=90); print("saved", out); sys.exit()
dors = [i for i, n in enumerate(mcols) if n.startswith("MD")]; vent = [i for i, n in enumerate(mcols) if n.startswith("MV")]
print("muscle mean mV  dorsal / ventral:", M[:, dors].mean(), M[:, vent].mean())

fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
for g in ["AVB", "AVA", "PVC"]:
    for i in groups[g]: ax[0].plot(t, V[:, i], label=ncols[i])
ax[0].legend(ncol=3, fontsize=8); ax[0].set_ylabel("command mV")
for g, c in [("DB", "tab:blue"), ("VB", "tab:cyan"), ("DA", "tab:red"), ("VA", "tab:orange")]:
    ax[1].plot(t, V[:, groups[g]].mean(1), color=c, label=g + " mean")
ax[1].legend(ncol=4, fontsize=8); ax[1].set_ylabel("motor mV")
ax[2].imshow(V[:, sorted(sum([groups[g] for g in ["DB","VB","DA","VA"]], []))].T, aspect="auto", cmap="viridis",
             extent=[t[0], t[-1], 0, 1]); ax[2].set_ylabel("motor neurons")
rows = lambda names: np.array([M[:, [i for i, n in enumerate(mcols) if n.startswith(p) and int(n[3:]) == k]].mean(1)
                               for k in range(1, 25) for p in [names]]).squeeze()
D = np.array([M[:, [i for i, n in enumerate(mcols) if n[:2] == "MD" and int(n[3:]) == k]].mean(1) for k in range(1, 25)])
Vv = np.array([M[:, [i for i, n in enumerate(mcols) if n[:2] == "MV" and int(n[3:]) == k]].mean(1) for k in range(1, 24)])
ax[3].imshow(D - np.vstack([Vv, Vv[-1:]]), aspect="auto", cmap="RdBu_r", extent=[t[0], t[-1], 24, 1]); ax[3].set_ylabel("muscle row (D-V mV)")
ax[3].set_xlabel("ms")
for a in ax: 
    a.axvspan(0, 400, color="b", alpha=0.05); a.axvspan(500, 1000, color="r", alpha=0.05)
out = os.path.join(run_dir, f"{ref}.png"); fig.tight_layout(); fig.savefig(out, dpi=90); print("saved", out)
