"""전역 상향 상태(runaway) 특성화: 자극 세기별 네트워크 평균 막전위와 A/B형 운동뉴런 분화의 시간 경과.
사용: uv run python experiments/phase1_runaway.py <net.nml> [--tag x]
"""
import os, re, sys, argparse, time
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network
from worm.neural import jaxsim

ap = argparse.ArgumentParser(); ap.add_argument("nml"); ap.add_argument("--tag", default="default")
ap.add_argument("--dt", type=float, default=0.25); ap.add_argument("--dur", type=float, default=4000)
ap.add_argument("--amps", default="0.5,1,2,3,5,10"); ap.add_argument("--stim", default="AVBL,AVBR"); ap.add_argument("--variant", default="V0")
ap.add_argument("--theta", default=None)
a = ap.parse_args()
from worm.neural.variants import make_variant, apply_theta
net = load_network(a.nml); net.pulses = np.zeros((0, 4)); net, vinfo = make_variant(net, a.variant); print("variant:", vinfo)
if a.theta:
    import json; th = json.load(open(a.theta))[-1]["theta"] if a.theta.endswith(".json") else [float(x) for x in a.theta.split(",")]
    net = apply_theta(net, th); a.variant += "-fit"; print("theta:", np.round(th, 3).tolist())
a.tag = f"{a.tag}_{a.variant}"; P = jaxsim.build_params(net)
isM = net.is_muscle(); names = net.names
grp = lambda pat: [i for i, n in enumerate(names) if re.fullmatch(pat, n)]
A_idx, B_idx = grp(r"[DV]A\d+"), grp(r"[DV]B\d+"); neu = np.where(~isM)[0]
stim_cells = [net.index(s) for s in a.stim.split(",")]
amps = [float(x) for x in a.amps.split(",")]
steps = int(round(a.dur / a.dt)); t = np.arange(steps) * a.dt
rec = 10; tr = t[::rec]
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True); rows = []
for amp in amps:
    I = np.zeros((steps, net.n)); I[:, stim_cells] = amp
    t0 = time.time(); _, Vs = jaxsim.simulate(P, a.dt, a.dur, net.n, I_ext=jnp.asarray(I), record_every=rec); Vs = np.asarray(Vs)
    mean_neu = Vs[:, neu].mean(1); dBA = Vs[:, B_idx].mean(1) - Vs[:, A_idx].mean(1)
    frac_up = (Vs[:, neu] > -30).mean(1)
    ax[0].plot(tr, mean_neu, label=f"{amp} pA"); ax[1].plot(tr, dBA); ax[2].plot(tr, frac_up)
    # 상향 상태 진입 시각: 뉴런의 50%가 −30 mV 초과
    up = np.where(frac_up > 0.5)[0]; t_up = tr[up[0]] if len(up) else None
    rows.append((amp, mean_neu[-1], dBA[:int(400/a.dt/rec)].max(), dBA[-1], t_up, frac_up[-1]))
    print(f"[{time.time()-t0:4.1f}s] amp {amp:5.1f} pA: final mean {mean_neu[-1]:6.1f} mV, B−A early max {rows[-1][2]:+.1f}, final {dBA[-1]:+.1f}, up-state at {t_up} ms, frac up {frac_up[-1]:.2f}")
ax[0].set_ylabel("mean neuron V (mV)"); ax[0].legend(ncol=3, fontsize=8); ax[1].set_ylabel("B − A motor (mV)"); ax[1].axhline(0, color="k", lw=0.5)
ax[2].set_ylabel("fraction neurons > −30 mV"); ax[2].set_xlabel("ms"); fig.suptitle(f"sustained stim of {a.stim} ({a.tag})"); fig.tight_layout()
os.makedirs("runs/phase1", exist_ok=True); out = f"runs/phase1/runaway_{a.tag}_{a.stim.split(',')[0]}"; fig.savefig(out + ".png", dpi=90)
md = f"### 지속 자극 {a.stim} ({a.tag}, dt {a.dt}, {a.dur:.0f} ms)\n\n| pA | 최종 뉴런 평균 (mV) | B−A 초기 최대 (mV) | B−A 최종 (mV) | 상향 상태 진입 (ms) | 최종 상향 비율 |\n|---|---|---|---|---|---|\n"
md += "\n".join(f"| {r[0]} | {r[1]:.1f} | {r[2]:+.1f} | {r[3]:+.1f} | {r[4] if r[4] is not None else '없음'} | {r[5]:.2f} |" for r in rows) + "\n"
open(out + ".md", "w").write(md); print(md); print("saved", out)
