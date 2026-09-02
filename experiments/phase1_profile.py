"""스텝 함수 구성 요소별 비용 측정 (2000 스텝)."""
import os, sys, time; sys.path.insert(0, os.getcwd())
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from worm.neural.connectome import load_network
from worm.neural import jaxsim
net = load_network("runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
for dtype in [jnp.float32]:
    P = jaxsim.build_params(net, dtype); S = jaxsim.init_state(P); N = net.n
    steps = 2000; ts = jnp.arange(steps) * 0.05; I = jnp.zeros((steps, N), dtype)
    def timeit(name, body):
        f = jax.jit(lambda S, xs: lax.scan(body, S, xs, unroll=jaxsim.UNROLL))
        f(S, (ts, I))[1].block_until_ready(); t0 = time.time(); f(S, (ts, I))[1].block_until_ready(); dt = time.time() - t0
        print(f"{name:28s} {dt/steps*1e6:7.1f} us/step")
    full = jaxsim.make_step(P, 0.05, N); timeit("full step", full)
    def only_gates(S, x):
        t, I_ext = x; Vn = S.V + 0.001 * I_ext
        g = lambda v, gg: v + (jaxsim._xinf(Vn, gg) - v) * (1 - jnp.exp(-0.05 / gg[0]))
        n, p, q, e, f = (g(getattr(S, k), P.gate[k]) for k in "npqef")
        return S._replace(V=Vn, n=n, p=p, q=q, e=e, f=f), Vn
    timeit("gates only", only_gates)
    def only_syn(S, x):
        t, I_ext = x; Gs = P.syn_wg * S.s; Gs_in = Gs[P.syn_in]; Gsyn = Gs_in.sum(1); GsynE = (Gs_in * P.syn_erev[P.syn_in]).sum(1)
        Vn = S.V + 0.001 * (GsynE - Gsyn * S.V)
        s_inf = 1 / (1 + jnp.exp((P.syn_vth - Vn[P.syn_pre]) / P.syn_delta)); s = S.s + (s_inf - S.s) * 0.01
        return S._replace(V=Vn, s=s), Vn
    timeit("synapses only", only_syn)
    def only_gj(S, x):
        t, I_ext = x; GgjV = (P.gj_wg_pad * S.V[P.gj_peer]).sum(1); Vn = S.V + 0.001 * (GgjV - P.gj_rowsum * S.V)
        return S._replace(V=Vn), Vn
    timeit("gap junctions only", only_gj)
    def trivial(S, x):
        t, I_ext = x; Vn = S.V + 0.001 * I_ext
        return S._replace(V=Vn), Vn
    timeit("trivial (loop overhead)", trivial)
