"""Phase 2b-3: 느린 전압 의존 K 전류(C2 확장)를 가진 고립 뉴런의 진동 탐색.
조합: g_Kw (nS) × V_half (mV) × τ_w (ms) × 정전류 (pA). 목표 0.3–1 Hz."""
import os, sys, itertools, numpy as np, jax, jax.numpy as jnp; sys.path.insert(0, os.getcwd())
jax.config.update("jax_enable_x64", True)
from worm.neural.connectome import load_network, ablate
from worm.neural import jaxsim
net = load_network("runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"); net.pulses = np.zeros((0, 4))
iso = ablate(net, net.names); P = jaxsim.build_params(iso); N = iso.n; neu = np.where(~iso.is_muscle())[0]
grid = list(itertools.product([1.0, 4.0], [-25.0, -15.0, -5.0], [2.0, 3.0, 4.0], [200.0, 500.0, 1000.0], [2.0, 5.0]))   # 108: g, Vhalf, k, tau, I
gK = np.array(P.gKw); vh = np.array(P.w_vhalf); wk = np.array(P.w_k); tw = np.array(P.w_tau); I = np.zeros(N)
for k, (g, v, kk, tau, i) in enumerate(grid):
    n = neu[k]; gK[n] = g; vh[n] = v; wk[n] = kk; tw[n] = tau; I[n] = i
P = P._replace(gKw=jnp.asarray(gK), w_vhalf=jnp.asarray(vh), w_k=jnp.asarray(wk), w_tau=jnp.asarray(tw))
dt, dur = 0.05, 12000.0; steps = int(dur / dt)
_, Vs = jaxsim.simulate(P, dt, dur, N, I_ext=jnp.broadcast_to(jnp.asarray(I), (steps, N)), record_every=20); Vs = np.asarray(Vs)
late = Vs[int(6000 / dt / 20):]; fs = 1000 / (dt * 20); rows = []
for k, g in enumerate(grid):
    v = late[:, neu[k]]; sd = v.std(); f = np.fft.rfftfreq(len(v), 1 / fs); Pw = np.abs(np.fft.rfft(v - v.mean())) ** 2; fpk = f[Pw[1:].argmax() + 1]
    rows.append((g, v.mean(), v.min(), v.max(), sd, fpk))
osc = [r for r in rows if r[4] > 2.0]; print(f"oscillating combos (std > 2 mV): {len(osc)} / {len(grid)}")
slow = [r for r in osc if 0.2 <= r[5] <= 1.5]; print(f"  with 0.2–1.5 Hz: {len(slow)}")
for g, m, lo, hi, sd, fpk in sorted(osc, key=lambda r: abs(np.log(r[5] / 0.5)))[:30]:
    print(f"  gKw {g[0]:3} nS, Vhalf {g[1]:5.0f}, k {g[2]:3.0f}, tau {g[3]:5.0f} ms, I {g[4]:3.0f} pA: V [{lo:6.1f},{hi:6.1f}] std {sd:5.1f} peak {fpk:5.2f} Hz")
os.makedirs("runs/phase2", exist_ok=True)
with open("runs/phase2/slowk_scan.md", "w") as fh:
    fh.write("| g_Kw (nS) | V_half (mV) | k (mV) | tau_w (ms) | I (pA) | V min | V max | std | 주파수 (Hz) |\n|---|---|---|---|---|---|---|---|---|\n")
    for g, m, lo, hi, sd, fpk in rows: fh.write(f"| {g[0]} | {g[1]:.0f} | {g[2]:.0f} | {g[3]:.0f} | {g[4]:.0f} | {lo:.1f} | {hi:.1f} | {sd:.2f} | {fpk:.2f} |\n")
