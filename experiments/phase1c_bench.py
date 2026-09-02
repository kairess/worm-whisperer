"""Phase 1c 타당성: vmap 배치 순방향과 역방향(grad) 비용 측정."""
import os, sys, time, numpy as np, jax, jax.numpy as jnp
from jax import lax
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network
from worm.neural.variants import make_variant
from worm.neural import jaxsim
net = load_network("runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"); net.pulses = np.zeros((0, 4))
net, _ = make_variant(net, "V1-split"); N = net.n
P = jaxsim.build_params(net, jnp.float32)
dt, dur, B = 0.5, 1500.0, 16
steps = int(dur / dt); ts = jnp.arange(steps, dtype=jnp.float32) * dt
stim_idx = jnp.arange(B); onehot = jax.nn.one_hot(stim_idx, N)          # B 자극 뉴런
I_t = (ts < 500).astype(jnp.float32)                                     # 500 ms 펄스

def forward(theta, P):
    # theta: 전역 파라미터 log-scale (chem, gj, leak)
    P2 = P._replace(syn_wg=P.syn_wg * jnp.exp(theta[0]), gj_wg_pad=P.gj_wg_pad * jnp.exp(theta[1]), gj_rowsum=P.gj_rowsum * jnp.exp(theta[1]), gL=P.gL * jnp.exp(theta[2]))
    step = jaxsim.make_step(P2, dt, N)
    def one(oh):
        S = jaxsim.init_state(P2)
        def body(S, t):
            S, V = step(S, (t, 2.0 * oh * I_t[jnp.int32(t / dt)]))
            return S, V
        body = jax.checkpoint(body)
        S, Vs = lax.scan(body, S, ts)
        return Vs.mean(0)                 # (N,) 평균 전압
    return jax.vmap(one)(onehot)          # (B, N)
def loss(theta, P):
    dV = forward(theta, P)
    return jnp.mean(dV ** 2)
theta = jnp.zeros(3)
f = jax.jit(forward); t0 = time.time(); out = f(theta, P); out.block_until_ready(); print(f"forward compile+run B={B}: {time.time()-t0:.1f}s")
t0 = time.time(); out = f(theta, P); out.block_until_ready(); tf = time.time() - t0; print(f"forward B={B}, {steps} steps: {tf:.1f}s → {tf/B:.2f}s per stim")
g = jax.jit(jax.grad(loss)); t0 = time.time(); gg = g(theta, P); gg.block_until_ready(); print(f"grad compile+run: {time.time()-t0:.1f}s")
t0 = time.time(); gg = g(theta, P); gg.block_until_ready(); print(f"grad B={B}: {time.time()-t0:.1f}s, grad = {np.asarray(gg)}")
