"""Phase 2b-2: Ca 활성화 K 전류(C1 확장)를 가진 고립 뉴런의 진동 파라미터 탐색.
조합: g_KCa (nS) × Ca 반활성 (mM) × Ca 붕괴 시간 (ms) × 정전류 (pA)."""
import os, sys, itertools, numpy as np, jax, jax.numpy as jnp; sys.path.insert(0, os.getcwd())
jax.config.update("jax_enable_x64", True)
from worm.neural.connectome import load_network, ablate
from worm.neural import jaxsim
net = load_network("runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"); net.pulses = np.zeros((0, 4))
iso = ablate(net, net.names); P = jaxsim.build_params(iso); N = iso.n; neu = np.where(~iso.is_muscle())[0]
grid = list(itertools.product([0.3, 1.0, 3.0, 10.0], [1e-7, 3e-7, 1e-6], [50.0, 200.0, 800.0], [3.0, 10.0]))   # 72
gK = np.array(P.gKCa); kh = np.array(P.kca_half); ct = np.array(P.ca_tau); I = np.zeros(N)
for k, (g, h, tau, i) in enumerate(grid):
    n = neu[k]; gK[n] = g; kh[n] = h; ct[n] = tau; I[n] = i
P = P._replace(gKCa=jnp.asarray(gK), kca_half=jnp.asarray(kh), ca_tau=jnp.asarray(ct))
dt, dur = 0.05, 10000.0; steps = int(dur / dt)
_, Vs = jaxsim.simulate(P, dt, dur, N, I_ext=jnp.broadcast_to(jnp.asarray(I), (steps, N)), record_every=20); Vs = np.asarray(Vs)
late = Vs[int(5000 / dt / 20):]; fs = 1000 / (dt * 20); rows = []
for k, g in enumerate(grid):
    v = late[:, neu[k]]; sd = v.std(); f = np.fft.rfftfreq(len(v), 1 / fs); Pw = np.abs(np.fft.rfft(v - v.mean())) ** 2; fpk = f[Pw[1:].argmax() + 1]
    rows.append((g, v.mean(), v.min(), v.max(), sd, fpk))
osc = [r for r in rows if r[4] > 2.0]; print(f"oscillating combos (std > 2 mV): {len(osc)} / {len(grid)}")
for g, m, lo, hi, sd, fpk in sorted(osc, key=lambda r: abs(np.log(r[5] / 0.5)))[:25]:
    print(f"  gKCa {g[0]:4} nS, Ca_half {g[1]:.0e}, tau_Ca {g[2]:4.0f} ms, I {g[3]:3.0f} pA: V [{lo:6.1f},{hi:6.1f}] std {sd:5.1f} peak {fpk:5.2f} Hz")
os.makedirs("runs/phase2", exist_ok=True)
with open("runs/phase2/kca_scan.md", "w") as fh:
    fh.write("| g_KCa (nS) | Ca_half (mM) | tau_Ca (ms) | I (pA) | V min | V max | std | 주파수 (Hz) |\n|---|---|---|---|---|---|---|---|\n")
    for g, m, lo, hi, sd, fpk in rows: fh.write(f"| {g[0]} | {g[1]:.0e} | {g[2]:.0f} | {g[3]:.0f} | {lo:.1f} | {hi:.1f} | {sd:.2f} | {fpk:.2f} |\n")
