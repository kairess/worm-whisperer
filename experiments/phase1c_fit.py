"""Phase 1c: 전역 동역학 파라미터를 Randi 2023 신호전파 아틀라스에 경사 기반으로 적합한다.
커넥톰(연결 존재/개수)과 세포 모델 구조는 고정. 학습 파라미터(6개):
  log g_exc, log g_inh (뉴런→뉴런 화학 시냅스 전도도 배율), log g_gj (갭정션 배율), log g_leak (뉴런 누설 배율),
  Vth_nn (화학 시냅스 반활성 전압, mV), log δ_nn (기울기 배율)
손실 = −Pearson(모델 ΔV, 실측 ΔF/F; 학습 자극 뉴런의 측정 쌍) + λ·상향상태 페널티
학습/검증 분할: 자극 뉴런 172개 중 32개 학습, 나머지 검증 (seed 0). 검증은 phase1b_sigprop.py 로 별도 수행.
사용: uv run python experiments/phase1c_fit.py --steps 60 --out runs/phase1c/fit_A
"""
import os, sys, json, time, argparse, numpy as np
import jax, jax.numpy as jnp
from jax import lax
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network
from worm.neural.variants import make_variant
from worm.neural import jaxsim

ap = argparse.ArgumentParser()
ap.add_argument("--nml", default="runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
ap.add_argument("--base", default="V1-split"); ap.add_argument("--out", default="runs/phase1c/fit")
ap.add_argument("--steps", type=int, default=60); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--B", type=int, default=32); ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--dur", type=float, default=1500)
ap.add_argument("--amp", type=float, default=2.0); ap.add_argument("--lam_up", type=float, default=1.0)
ap.add_argument("--lam_auc", type=float, default=1.0); ap.add_argument("--lam_r", type=float, default=1.0); ap.add_argument("--n_neg", type=int, default=64)
ap.add_argument("--lam_inh", type=float, default=0.0, help="실측 억제 쌍(q<0.05, dff<0)에서 모델 ΔV>0 페널티")
ap.add_argument("--lam_circuit", type=float, default=0.0, help="Chalfie 1985: PLM→(AVB>AVA), ALM/AVM→(AVA>AVB) 힌지 손실")
ap.add_argument("--init", default="0,0,-1.2,2.3,-40,0", help="log g_exc, log g_inh, log g_gj, log g_leak, Vth, log δ")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

net = load_network(a.nml); net.pulses = np.zeros((0, 4)); net, vinfo = make_variant(net, a.base); N = net.n; isM = net.is_muscle()
P0 = jaxsim.build_params(net, jnp.float32)
nn = ~(isM[net.syn_pre] | isM[net.syn_post]); exc = nn & np.array(["exc" in s for s in net.syn_id]); inh = nn & np.array(["inh" in s for s in net.syn_id])
m_exc = jnp.asarray(np.append(exc, False)); m_inh = jnp.asarray(np.append(inh, False)); m_nn = jnp.asarray(np.append(nn, False)); m_neu = jnp.asarray(~isM)

atl = np.load("worm/data/randi2023_sigprop.npz"); ids = [str(x) for x in atl["ids"]]; dff, q = atl["dff"], atl["q"]
aidx = {n: i for i, n in enumerate(ids)}; aidx["AWCL"] = aidx["AWCOFF"]; aidx["AWCR"] = aidx["AWCON"]
model_of_atlas = {i: net.index(n) for n, i in aidx.items() if n in net.names}
meas = np.isfinite(dff).sum(0); stims = [j for j in np.argsort(-meas) if meas[j] >= 30 and j in model_of_atlas]
rng = np.random.default_rng(0); train = sorted(rng.choice(stims, a.B, replace=False).tolist()); test = [j for j in stims if j not in train]
json.dump({"train": [ids[j] for j in train], "test": [ids[j] for j in test]}, open(f"{a.out}/split.json", "w"), indent=1)
# 학습 타깃 행렬 (B, N): 측정 마스크와 dff
Mk = np.zeros((a.B, N), np.float32); Y = np.zeros((a.B, N), np.float32); stim_model = np.zeros(a.B, np.int32)
for b, j in enumerate(train):
    stim_model[b] = model_of_atlas[j]
    for i in range(len(ids)):
        if np.isfinite(dff[i, j]) and i != j and i in model_of_atlas:
            Mk[b, model_of_atlas[i]] = 1; Y[b, model_of_atlas[i]] = dff[i, j]
# 소프트 AUROC 용 (유의 쌍, 비유의 쌍) 표본: 유의 쌍마다 같은 자극의 비유의 측정 쌍 n_neg 개
Sg = np.zeros((a.B, N), bool)
for b, j in enumerate(train):
    for i in range(len(ids)):
        if np.isfinite(dff[i, j]) and i != j and i in model_of_atlas and q[i, j] < 0.05: Sg[b, model_of_atlas[i]] = True
pos_b, pos_i, neg_b, neg_i = [], [], [], []
for b in range(a.B):
    P_ = np.where(Sg[b])[0]; Nn = np.where((Mk[b] > 0) & ~Sg[b])[0]
    for i in P_:
        nn_ = rng.choice(Nn, a.n_neg, replace=len(Nn) < a.n_neg)
        pos_b += [b] * a.n_neg; pos_i += [i] * a.n_neg; neg_b += [b] * a.n_neg; neg_i += nn_.tolist()
pos_b, pos_i, neg_b, neg_i = (jnp.asarray(v, jnp.int32) for v in (pos_b, pos_i, neg_b, neg_i))
inh_b, inh_i = np.where(Sg & (np.asarray(Y) < 0)); inh_b, inh_i = jnp.asarray(inh_b, jnp.int32), jnp.asarray(inh_i, jnp.int32); print(f"train inhibitory pairs: {len(inh_b)}")
ci = {n: net.index(n) for n in ["PLML", "PLMR", "ALML", "ALMR", "AVM", "AVBL", "AVBR", "AVAL", "AVAR"]}
oh_plm = jnp.zeros(N).at[jnp.array([ci["PLML"], ci["PLMR"]])].set(1.0); oh_alm = jnp.zeros(N).at[jnp.array([ci["ALML"], ci["ALMR"], ci["AVM"]])].set(1.0)
AVB_i = jnp.array([ci["AVBL"], ci["AVBR"]]); AVA_i = jnp.array([ci["AVAL"], ci["AVAR"]])
print(f"significant train pairs: {int(Sg.sum())}, ranking pairs: {len(pos_b)}")
Mk, Y = jnp.asarray(Mk), jnp.asarray(Y); onehot = jax.nn.one_hot(jnp.asarray(stim_model), N)
steps = int(a.dur / a.dt); ts = jnp.arange(steps, dtype=jnp.float32) * a.dt; I_t = (ts < 500).astype(jnp.float32) * a.amp
print(f"base {vinfo}; train {a.B} stims, test {len(test)}; measured pairs in train: {int(Mk.sum())}; steps {steps}")

def apply(theta, P):
    le, li, lg, ll, vth, ld = theta
    wg = P.syn_wg * jnp.where(m_exc, jnp.exp(le), 1.0) * jnp.where(m_inh, jnp.exp(li), 1.0)
    return P._replace(syn_wg=wg, gj_wg_pad=P.gj_wg_pad * jnp.exp(lg), gj_rowsum=P.gj_rowsum * jnp.exp(lg),
                      gL=P.gL * jnp.where(m_neu, jnp.exp(ll), 1.0), syn_vth=jnp.where(m_nn, vth, P.syn_vth),
                      syn_delta=P.syn_delta * jnp.where(m_nn, jnp.exp(ld), 1.0))

def forward(theta):
    P = apply(theta, P0); step = jaxsim.make_step(P, a.dt, N); S0 = jaxsim.init_state(P)
    def one(oh):
        def body(carry, t):
            S, acc = carry
            S, V = step(S, (t, oh * I_t[jnp.int32(t / a.dt)]))
            return (S, acc + V), None
        body = jax.checkpoint(body)
        (S, acc), _ = lax.scan(body, (S0, jnp.zeros(N)), ts)
        return acc / steps, S.V
    meanV, Vend = jax.vmap(one)(onehot)
    circ = jax.vmap(one)(jnp.stack([oh_plm, oh_alm]))[0] if a.lam_circuit > 0 else None
    # 대조군 (무자극)
    def ctrl():
        def body(carry, t):
            S, acc = carry; S, V = step(S, (t, jnp.zeros(N))); return (S, acc + V), None
        body = jax.checkpoint(body)
        (S, acc), _ = lax.scan(body, (S0, jnp.zeros(N)), ts); return acc / steps
    c = ctrl()
    return meanV - c[None, :], Vend, (None if circ is None else circ - c[None, :])

def loss_fn(theta):
    dV, Vend, circ = forward(theta)
    w = Mk; x = dV; y = Y
    mx = (w * x).sum() / w.sum(); my = (w * y).sum() / w.sum()
    cov = (w * (x - mx) * (y - my)).sum(); vx = (w * (x - mx) ** 2).sum(); vy = (w * (y - my) ** 2).sum()
    r = cov / jnp.sqrt(vx * vy + 1e-12)
    up = jax.nn.sigmoid((Vend[:, m_neu] + 30.0) / 5.0).mean()        # 상향 상태 소프트 비율
    z = jnp.abs(x) / ((w * jnp.abs(x)).sum() / w.sum() + 1e-9)          # 스케일 정규화한 |ΔV|
    auc_soft = jax.nn.sigmoid((z[pos_b, pos_i] - z[neg_b, neg_i]) / 0.5).mean()
    L = -a.lam_r * r - a.lam_auc * auc_soft + a.lam_up * up
    inh = jax.nn.softplus(x[inh_b, inh_i] / 1.0).mean() if len(inh_b) else 0.0     # 억제 쌍에서 ΔV>0 이면 페널티 (mV 단위)
    circ_pen = 0.0
    if circ is not None:
        d_plm = circ[0][AVB_i].mean() - circ[0][AVA_i].mean(); d_alm = circ[1][AVA_i].mean() - circ[1][AVB_i].mean()
        circ_pen = jax.nn.relu(2.0 - d_plm) + jax.nn.relu(2.0 - d_alm)               # 2 mV 마진 힌지
    L = L + a.lam_inh * inh + a.lam_circuit * circ_pen
    return L, (r, auc_soft, up, inh, circ_pen)

theta = jnp.asarray([float(x) for x in a.init.split(",")], jnp.float32)
vg = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
import optax
opt = optax.adam(a.lr); state = opt.init(theta); log = []
for it in range(a.steps):
    t0 = time.time(); (L, (r, auc, up, inh, cp)), g = vg(theta); g = jnp.nan_to_num(g)
    upd, state = opt.update(g, state); theta = optax.apply_updates(theta, upd)
    theta = theta.at[4].set(jnp.clip(theta[4], -70.0, 10.0))
    rec = {"it": it, "loss": float(L), "pearson": float(r), "auc_soft": float(auc), "up": float(up), "inh": float(inh), "circuit": float(cp), "theta": [float(x) for x in theta], "grad": [float(x) for x in g], "sec": time.time() - t0}
    log.append(rec); json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    print(f"it {it:3d} loss {L:+.4f} r {r:+.4f} aucS {auc:.3f} up {up:.3f} inh {inh:.3f} circ {cp:.2f} theta {np.round(np.asarray(theta), 3).tolist()} ({rec['sec']:.0f}s)", flush=True)
print("done")
