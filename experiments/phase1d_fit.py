"""Phase 1d: 연결 클래스별 파라미터(θ 63개, worm/neural/variants.py 의 connection_classes)를 아틀라스 + 억제 + 터치 회로 목표에 적합한다.

phase1c_fit.py 와 같은 프로토콜(학습 자극 뉴런 32개, seed 0 분할, dt 0.5 ms, 1.5 s, 2 pA 500 ms)과 손실에
  + λ_inh · [실측 억제 쌍(학습 세트)에서 모델 ΔV>0 페널티 + 실측 흥분 유의 쌍에서 ΔV<0 페널티] (softplus, mV; 부호 균형)
  + λ_circuit · Chalfie 1985 힌지 (PLM 자극: AVB−AVA ≥ margin, ALM+AVM 자극: AVA−AVB ≥ margin; 5 pA 지속, 마지막 500 ms 평균)
  + λ_prior · ‖θ − θ_init‖² / n  (클래스 파라미터가 초기 전역값에서 멀어지는 것에 대한 능형 페널티)
초기값: Vfit(θ6) + 억제 동작점(--init_inh "Vth_inh,log g_inh,log δ_inh"; phase1d_inh_scan.py 결과로 정함).
사용: uv run python experiments/phase1d_fit.py --steps 60 --out runs/phase1d/fit_E --init_inh=-50,-2.3,0
검증: phase1b_sigprop.py --theta_class runs/phase1d/fit_E/log.json --exclude runs/phase1d/fit_E/split.json
"""
import os, sys, json, time, argparse, numpy as np
import jax, jax.numpy as jnp
from jax import lax
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network
from worm.neural.variants import make_variant, THETA_FIT_C, connection_classes, theta_class_from_theta8, N_SYN_CLS, N_GJ_CLS, N_CELL_CLS, syn_class_name, gj_class_name
from worm.neural import jaxsim

ap = argparse.ArgumentParser()
ap.add_argument("--nml", default="runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
ap.add_argument("--base", default="V1-split"); ap.add_argument("--out", default="runs/phase1d/fit")
ap.add_argument("--steps", type=int, default=60); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--B", type=int, default=32); ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--dur", type=float, default=1500)
ap.add_argument("--amp", type=float, default=2.0); ap.add_argument("--lam_up", type=float, default=1.0)
ap.add_argument("--lam_auc", type=float, default=1.0); ap.add_argument("--lam_r", type=float, default=1.0); ap.add_argument("--n_neg", type=int, default=64)
ap.add_argument("--lam_inh", type=float, default=0.3); ap.add_argument("--lam_circuit", type=float, default=0.3); ap.add_argument("--margin", type=float, default=2.0)
ap.add_argument("--lam_prior", type=float, default=0.01)
ap.add_argument("--init_inh", default="-50,-2.3,0", help="Vth_inh(mV), log g_inh(0.29 nS 대비), log δ_inh")
ap.add_argument("--init_theta", default=None, help="log.json 경로: 마지막 스텝의 θ(63) 에서 시작")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

net = load_network(a.nml); net.pulses = np.zeros((0, 4)); net, vinfo = make_variant(net, a.base); N = net.n; isM = net.is_muscle()
P0 = jaxsim.build_params(net, jnp.float32)
sc, gc, cc = connection_classes(net)
# Params 수준 인덱스: 시냅스는 더미(마지막) 포함, 갭정션은 패딩된 (N,D) 표에 맞춘다
syn_cls = jnp.asarray(np.append(sc, N_SYN_CLS)); syn_fixed = syn_cls >= N_SYN_CLS
ends = np.concatenate([net.gj_a, net.gj_b]); gcls2 = np.concatenate([gc, gc])
gj_lists = jaxsim._padded_lists(ends, N, fill=-1) if len(ends) else np.full((N, 1), -1)
gj_cls_pad = jnp.asarray(np.where(gj_lists >= 0, gcls2[np.maximum(gj_lists, 0)], N_GJ_CLS)); gj_fixed = gj_cls_pad >= N_GJ_CLS
cell_cls = jnp.asarray(cc); cell_fixed = cell_cls >= N_CELL_CLS
m_neu = jnp.asarray(~isM)

atl = np.load("worm/data/randi2023_sigprop.npz"); ids = [str(x) for x in atl["ids"]]; dff, q = atl["dff"], atl["q"]
aidx = {n: i for i, n in enumerate(ids)}; aidx["AWCL"] = aidx["AWCOFF"]; aidx["AWCR"] = aidx["AWCON"]
model_of_atlas = {i: net.index(n) for n, i in aidx.items() if n in net.names}
meas = np.isfinite(dff).sum(0); stims = [j for j in np.argsort(-meas) if meas[j] >= 30 and j in model_of_atlas]
rng = np.random.default_rng(0); train = sorted(rng.choice(stims, a.B, replace=False).tolist()); test = [j for j in stims if j not in train]
json.dump({"train": [ids[j] for j in train], "test": [ids[j] for j in test]}, open(f"{a.out}/split.json", "w"), indent=1)
Mk = np.zeros((a.B, N), np.float32); Y = np.zeros((a.B, N), np.float32); stim_model = np.zeros(a.B, np.int32); Sg = np.zeros((a.B, N), bool)
for b, j in enumerate(train):
    stim_model[b] = model_of_atlas[j]
    for i in range(len(ids)):
        if np.isfinite(dff[i, j]) and i != j and i in model_of_atlas:
            Mk[b, model_of_atlas[i]] = 1; Y[b, model_of_atlas[i]] = dff[i, j]; Sg[b, model_of_atlas[i]] = q[i, j] < 0.05
pos_b, pos_i, neg_b, neg_i = [], [], [], []
for b in range(a.B):
    P_ = np.where(Sg[b])[0]; Nn = np.where((Mk[b] > 0) & ~Sg[b])[0]
    for i in P_:
        nn_ = rng.choice(Nn, a.n_neg, replace=len(Nn) < a.n_neg)
        pos_b += [b] * a.n_neg; pos_i += [i] * a.n_neg; neg_b += [b] * a.n_neg; neg_i += nn_.tolist()
pos_b, pos_i, neg_b, neg_i = (jnp.asarray(v, jnp.int32) for v in (pos_b, pos_i, neg_b, neg_i))
inh_b, inh_i = np.where(Sg & (Y < 0)); inh_b, inh_i = jnp.asarray(inh_b, jnp.int32), jnp.asarray(inh_i, jnp.int32)
exc_b, exc_i = np.where(Sg & (Y > 0)); exc_b, exc_i = jnp.asarray(exc_b, jnp.int32), jnp.asarray(exc_i, jnp.int32)   # 부호 균형: 흥분 유의 쌍에서 ΔV<0 페널티 (전역 음화 퇴화 방지)
ci = {n: net.index(n) for n in ["PLML", "PLMR", "ALML", "ALMR", "AVM", "AVBL", "AVBR", "AVAL", "AVAR"]}
oh_plm = jnp.zeros(N).at[jnp.array([ci["PLML"], ci["PLMR"]])].set(1.0); oh_alm = jnp.zeros(N).at[jnp.array([ci["ALML"], ci["ALMR"], ci["AVM"]])].set(1.0)
AVB_i = jnp.array([ci["AVBL"], ci["AVBR"]]); AVA_i = jnp.array([ci["AVAL"], ci["AVAR"]])
Mk, Y = jnp.asarray(Mk), jnp.asarray(Y); onehot = jax.nn.one_hot(jnp.asarray(stim_model), N)
steps = int(a.dur / a.dt); ts = jnp.arange(steps, dtype=jnp.float32) * a.dt; I_t = (ts < 500).astype(jnp.float32) * a.amp
circ_amp = 5.0; circ_from = int(1000 / a.dt)         # 회로 목표: 5 pA 지속, 1.0–1.5 s 평균
print(f"base {vinfo}; train {a.B} stims, test {len(test)}; measured pairs {int(Mk.sum())}; sig {int(Sg.sum())}; inh pairs {len(inh_b)}; rank pairs {len(pos_b)}; steps {steps}")

def apply(theta, P):
    g = jnp.concatenate([theta[:18], jnp.zeros(1)]); v = jnp.concatenate([theta[18:36], jnp.zeros(1)]); d = jnp.concatenate([theta[36:54], jnp.zeros(1)])
    gj = jnp.concatenate([theta[54:60], jnp.zeros(1)]); lk = jnp.concatenate([theta[60:63], jnp.zeros(1)])
    wg = P.syn_wg * jnp.exp(g[syn_cls]); vth = jnp.where(syn_fixed, P.syn_vth, v[syn_cls]); delta = P.syn_delta * jnp.exp(d[syn_cls])
    gjp = P.gj_wg_pad * jnp.exp(gj[gj_cls_pad])
    return P._replace(syn_wg=wg, syn_vth=vth, syn_delta=delta, gj_wg_pad=gjp, gj_rowsum=gjp.sum(1), gL=P.gL * jnp.exp(lk[cell_cls]))

def forward(theta):
    P = apply(theta, P0); step = jaxsim.make_step(P, a.dt, N); S0 = jaxsim.init_state(P)
    def one(oh, I_of_t, acc_from):
        def body(carry, k):
            S, acc = carry; t = k * a.dt
            S, V = step(S, (t, oh * I_of_t[k]))
            return (S, acc + jnp.where(k >= acc_from, V, 0.0)), None
        body = jax.checkpoint(body)
        (S, acc), _ = lax.scan(body, (S0, jnp.zeros(N)), jnp.arange(steps))
        return acc / (steps - acc_from), S.V
    meanV, Vend = jax.vmap(lambda oh: one(oh, I_t, 0))(onehot)
    c, _ = one(jnp.zeros(N), I_t, 0)
    circ = None
    if a.lam_circuit > 0:
        I_c = jnp.full(steps, circ_amp, jnp.float32)
        cm, _ = jax.vmap(lambda oh: one(oh, I_c, circ_from))(jnp.stack([oh_plm, oh_alm])); cc0, _ = one(jnp.zeros(N), I_c, circ_from)
        circ = cm - cc0[None, :]
    return meanV - c[None, :], Vend, circ

def loss_fn(theta, theta0):
    dV, Vend, circ = forward(theta)
    w = Mk; x = dV; y = Y
    mx = (w * x).sum() / w.sum(); my = (w * y).sum() / w.sum()
    cov = (w * (x - mx) * (y - my)).sum(); vx = (w * (x - mx) ** 2).sum(); vy = (w * (y - my) ** 2).sum()
    r = cov / jnp.sqrt(vx * vy + 1e-12)
    up = jax.nn.sigmoid((Vend[:, m_neu] + 30.0) / 5.0).mean()
    z = jnp.abs(x) / ((w * jnp.abs(x)).sum() / w.sum() + 1e-9)
    auc_soft = jax.nn.sigmoid((z[pos_b, pos_i] - z[neg_b, neg_i]) / 0.5).mean()
    inh = jax.nn.softplus(x[inh_b, inh_i] / 1.0).mean() if len(inh_b) else 0.0
    inh = inh + jax.nn.softplus(-x[exc_b, exc_i] / 1.0).mean()
    circ_pen = 0.0; d_plm = d_alm = 0.0
    if circ is not None:
        d_plm = circ[0][AVB_i].mean() - circ[0][AVA_i].mean(); d_alm = circ[1][AVA_i].mean() - circ[1][AVB_i].mean()
        circ_pen = jax.nn.relu(a.margin - d_plm) + jax.nn.relu(a.margin - d_alm)
    prior = ((theta - theta0) ** 2).mean()
    L = -a.lam_r * r - a.lam_auc * auc_soft + a.lam_up * up + a.lam_inh * inh + a.lam_circuit * circ_pen + a.lam_prior * prior
    return L, (r, auc_soft, up, inh, circ_pen, d_plm, d_alm, prior)

vth_i, lg_i, ld_i = [float(x) for x in a.init_inh.split(",")]
th8 = list(THETA_FIT_C); th8[1] = lg_i; th8 += [vth_i, ld_i]
theta0 = jnp.asarray(theta_class_from_theta8(th8), jnp.float32)
theta = jnp.asarray(json.load(open(a.init_theta))[-1]["theta"], jnp.float32) if a.init_theta else theta0
vg = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
import optax
opt = optax.adam(a.lr); state = opt.init(theta); log = []
names = [f"g:{syn_class_name(c)}" for c in range(18)] + [f"vth:{syn_class_name(c)}" for c in range(18)] + [f"ld:{syn_class_name(c)}" for c in range(18)] + [f"gj:{gj_class_name(c)}" for c in range(6)] + [f"leak:{t}" for t in ["S", "I", "M"]]
json.dump({"names": names, "theta0": [float(x) for x in theta0], "args": vars(a)}, open(f"{a.out}/meta.json", "w"), indent=1)
for it in range(a.steps):
    t0 = time.time(); (L, (r, auc, up, inh, cp, dp, da, pr)), g = vg(theta, theta0); g = jnp.nan_to_num(g)
    upd, state = opt.update(g, state); theta = optax.apply_updates(theta, upd)
    theta = theta.at[18:36].set(jnp.clip(theta[18:36], -70.0, 10.0))
    rec = {"it": it, "loss": float(L), "pearson": float(r), "auc_soft": float(auc), "up": float(up), "inh": float(inh), "circuit": float(cp), "d_plm": float(dp), "d_alm": float(da), "prior": float(pr),
           "theta": [float(x) for x in theta], "grad_norm": float(jnp.linalg.norm(g)), "sec": time.time() - t0}
    log.append(rec); json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    print(f"it {it:3d} loss {L:+.4f} r {r:+.4f} aucS {auc:.3f} up {up:.3f} inh {inh:.3f} circ {cp:.2f} (PLM {dp:+.1f} ALM {da:+.1f}) prior {pr:.3f} |g| {rec['grad_norm']:.2f} ({rec['sec']:.0f}s)", flush=True)
print("done")
