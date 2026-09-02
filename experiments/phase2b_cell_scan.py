"""Phase 2b-1: c302 C2 세포 모델의 채널 밀도 공간에서 고유 진동 해를 찾는다.
고립 뉴런 302개에 서로 다른 (g_Ca, g_Ks, g_L 배율, 정전류) 조합을 배정해 한 번에 시뮬레이션."""
import os, sys, itertools, numpy as np, jax, jax.numpy as jnp; sys.path.insert(0, os.getcwd())
jax.config.update("jax_enable_x64", True)
from worm.neural.connectome import load_network, ablate
from worm.neural import jaxsim
net = load_network("runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"); net.pulses = np.zeros((0, 4))
iso = ablate(net, net.names); P = jaxsim.build_params(iso); N = iso.n; neu = np.where(~iso.is_muscle())[0]
grid = list(itertools.product([0.3, 1, 3, 10], [0.3, 1, 3, 10], [1, 10, 100], [2.0, 5.0, 10.0]))   # 144
gCa = np.array(P.gCa); gKs = np.array(P.gKs); gL = np.array(P.gL); I = np.zeros(N)
for k, (a, b, c, i) in enumerate(grid):
    n = neu[k]; gCa[n] *= a; gKs[n] *= b; gL[n] *= c; I[n] = i
P = P._replace(gCa=jnp.asarray(gCa), gKs=jnp.asarray(gKs), gL=jnp.asarray(gL))
dt, dur = 0.05, 8000.0; steps = int(dur / dt)
_, Vs = jaxsim.simulate(P, dt, dur, N, I_ext=jnp.broadcast_to(jnp.asarray(I), (steps, N)), record_every=20); Vs = np.asarray(Vs)
late = Vs[int(4000 / dt / 20):]; fs = 1000 / (dt * 20)
rows = []
for k, g in enumerate(grid):
    v = late[:, neu[k]]; sd = v.std()
    f = np.fft.rfftfreq(len(v), 1 / fs); Pw = np.abs(np.fft.rfft(v - v.mean())) ** 2; fpk = f[Pw[1:].argmax() + 1]
    rows.append((g, v.mean(), v.min(), v.max(), sd, fpk))
osc = [r for r in rows if r[4] > 2.0]
print(f"oscillating combos (std > 2 mV): {len(osc)} / {len(grid)}")
for g, m, lo, hi, sd, fpk in sorted(osc, key=lambda r: -r[4])[:25]:
    print(f"  gCa×{g[0]:<4} gKs×{g[1]:<4} gL×{g[2]:<4} I {g[3]:4.0f} pA: V [{lo:7.1f},{hi:7.1f}] mean {m:7.1f} std {sd:5.1f} peak {fpk:5.2f} Hz")
os.makedirs("runs/phase2", exist_ok=True)
with open("runs/phase2/cell_scan.md", "w") as fh:
    fh.write("| gCa× | gKs× | gL× | I (pA) | V min | V max | std | 주파수 (Hz) |\n|---|---|---|---|---|---|---|---|\n")
    for g, m, lo, hi, sd, fpk in rows: fh.write(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]:.0f} | {lo:.1f} | {hi:.1f} | {sd:.2f} | {fpk:.2f} |\n")
