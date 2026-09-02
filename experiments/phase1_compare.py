"""Phase 1 검증: JAX 재구현 vs NEURON(c302 C2) 전압 궤적 비교.
사용: uv run python experiments/phase1_compare.py runs/phase0/c302_C2_LW_Full_avb-ava [--x32]
"""
import os, sys, time
import numpy as np
import jax
if "--x32" not in sys.argv:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network, lems_settings
from worm.neural import jaxsim

run_dir = sys.argv[1].rstrip("/"); ref = os.path.basename(run_dir)
net = load_network(os.path.join(run_dir, ref + ".net.nml"))
dt, duration, cols = lems_settings(os.path.join(run_dir, f"LEMS_{ref}.xml"))
print(f"net: {net.n} cells, chem {len(net.syn_pre)}, gj {len(net.gj_a)}; dt {dt} ms, duration {duration} ms")

dtype = jnp.float32 if "--x32" in sys.argv else jnp.float64
dt_ref = dt
if "--dt" in sys.argv:
    dt = float(sys.argv[sys.argv.index("--dt") + 1])
sub = int(round(dt / dt_ref)); assert abs(sub * dt_ref - dt) < 1e-9, "dt must be a multiple of reference dt"
P = jaxsim.build_params(net, dtype)
t0 = time.time(); S, Vs = jaxsim.simulate(P, dt, duration, net.n); Vs.block_until_ready(); t_first = time.time() - t0
t0 = time.time(); S, Vs = jaxsim.simulate(P, dt, duration, net.n); Vs.block_until_ready(); t_second = time.time() - t0
print(f"jax: first run (compile 포함) {t_first:.2f}s, second {t_second:.2f}s for {duration/1000:.1f}s sim → {t_second/(duration/1000):.2f} s per sim-second")
Vs = np.asarray(Vs)   # (steps, N) : V after each step, i.e. t = dt, 2dt, ...

# NEURON 출력: 행 = t=0, dt, ..., duration (steps+1 행), 열 = LEMS 컬럼 순서
def load(fname):
    d = np.loadtxt(os.path.join(run_dir, fname))[::sub]
    return d[:, 0] * 1000, d[:, 1:] * 1000, cols[fname]
tN, VN, names_n = load(f"{ref}.dat")
mfile = f"{ref}.muscles.dat"
if os.path.exists(os.path.join(run_dir, mfile)):
    _, VM, names_m = load(mfile); VN = np.hstack([VN, VM]); names_n = names_n + names_m
order = [net.index(n) for n in names_n]
VJ = np.vstack([np.asarray(P.v_init)[order][None, :], Vs[:, order]])   # t=0 행 추가
assert VJ.shape == VN.shape, (VJ.shape, VN.shape)
err = VJ - VN
rms = np.sqrt((err ** 2).mean(0)); mx = np.abs(err).max(0)
mus = np.array([n.startswith("M") and n[1] in "DV" for n in names_n])
print(f"dt {dt} ms: ", end="")
print(f"RMS over all cells/time: {np.sqrt((err**2).mean()):.4f} mV | neurons {np.sqrt((err[:, ~mus]**2).mean()):.4f} | muscles {np.sqrt((err[:, mus]**2).mean()):.4f}")
print(f"max |err|: {mx.max():.4f} mV ({names_n[mx.argmax()]}), cells with RMS>1mV: {(rms>1).sum()}")
worst = np.argsort(-rms)[:8]
print("worst cells:", [(names_n[i], round(float(rms[i]), 3)) for i in worst])
print("err vs time (RMS over cells) at 100ms steps:", [round(float(np.sqrt((err[int(k/dt)]**2).mean())), 3) for k in range(0, int(duration)+1, 100)])

show = ["AVBL", "AVAL", "DB1", "VB1", "DA1", "VA1", "MDL08", "MVL08"]
fig, ax = plt.subplots(len(show), 1, figsize=(10, 2.0 * len(show)), sharex=True)
for a, nm in zip(ax, show):
    if nm not in names_n: continue
    i = names_n.index(nm)
    a.plot(tN, VN[:, i], "k", lw=2, label="NEURON"); a.plot(tN, VJ[:, i], "r--", lw=1, label="JAX")
    a.set_ylabel(nm); a.text(0.01, 0.8, f"RMS {rms[i]:.3f} mV", transform=a.transAxes, fontsize=8)
ax[0].legend(); ax[-1].set_xlabel("ms"); fig.tight_layout()
out = os.path.join(run_dir, f"{ref}.jax_vs_neuron{'_x32' if '--x32' in sys.argv else ''}{'_dt'+str(dt) if '--dt' in sys.argv else ''}.png"); fig.savefig(out, dpi=90); print("saved", out)
