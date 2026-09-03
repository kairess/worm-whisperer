"""Phase 1d: 억제 시냅스의 동작점을 흥분과 분리했을 때(θ8) 억제 재현·안정성·터치 회로 방향을 탐색한다.

배경 (docs/RESULTS_PHASE1.md 6절): Vfit 은 억제 시냅스를 사실상 제거했고(g_inh ≈ 0), Vth 가 +1.5 mV 라 휴지(−53 mV)에서
화학 시냅스가 꺼져 있다. 여기서는 억제 시냅스만 별도의 Vth_inh(휴지 근처) / δ_inh / g_inh 를 갖게 하고, 흥분·갭정션·누설은 Vfit 값을 유지한다.

각 설정마다:
  (a) 안정성: PLM, AVA 양측 10 pA 4 s 지속 자극 후 상향 상태(> −30 mV) 뉴런 비율
  (b) 아틀라스: 실측 억제 쌍(q<0.05, ΔF/F<0)이 있는 자극 뉴런 전부(2 pA, 500 ms, 2 s 창)에서 억제 재현율(전체 / 커넥톰 경로가 있는 부분집합),
      흥분 재현율, AUROC (phase1b 와 같은 프로토콜)
  (c) 회로: PLM 5 pA, ALM+AVM 5 pA 2 s → 마지막 500 ms 의 AVB−AVA 평균 ΔV (Chalfie 1985: PLM 은 AVB>AVA, ALM 은 AVA>AVB)
사용: uv run python experiments/phase1d_inh_scan.py [--grid quick|full] [--out runs/phase1d/scan.json]
"""
import os, sys, json, time, argparse, itertools, numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network
from worm.neural.variants import make_variant, apply_theta, apply_theta_class, THETA_FIT_C
from worm.neural import jaxsim
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--nml", default="runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
ap.add_argument("--grid", default="quick"); ap.add_argument("--out", default="runs/phase1d/scan.json")
ap.add_argument("--dt", type=float, default=0.25); ap.add_argument("--configs", default=None, help="'vth,lg,ld;vth,lg,ld;...' 직접 지정")
ap.add_argument("--theta_class", default=None, help="phase1d_fit 의 log.json (마지막 스텝 θ63) 을 평가; --step 으로 특정 스텝")
ap.add_argument("--step", type=int, default=-1); ap.add_argument("--exclude", default=None, help="split.json: train 자극 뉴런 제외")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)

net0 = load_network(a.nml); net0.pulses = np.zeros((0, 4)); base, _ = make_variant(net0, "V1-split"); N = base.n; isM = base.is_muscle()
ix = {n: i for i, n in enumerate(base.names)}
atl = np.load("worm/data/randi2023_sigprop.npz"); ids = [str(x) for x in atl["ids"]]; dff, q = atl["dff"], atl["q"]
aidx = {n: i for i, n in enumerate(ids)}; aidx["AWCL"] = aidx["AWCOFF"]; aidx["AWCR"] = aidx["AWCON"]
moa = {i: base.index(n) for n, i in aidx.items() if n in base.names}
sig = (q < 0.05) & np.isfinite(dff); negp = sig & (dff < 0)
# 커넥톰 경로가 있는 억제 쌍: 직접 억제 시냅스, 또는 j→k(흥분/갭) →i(억제) 2단계
nn = ~(isM[base.syn_pre] | isM[base.syn_post]); inh = nn & np.array(["inh" in s for s in base.syn_id]); exc = nn & ~inh
Wi = np.zeros((N, N)); np.add.at(Wi, (base.syn_post[inh], base.syn_pre[inh]), base.syn_w[inh])
We = np.zeros((N, N)); np.add.at(We, (base.syn_post[exc], base.syn_pre[exc]), base.syn_w[exc])
Wg = np.zeros((N, N)); np.add.at(Wg, (base.gj_b, base.gj_a), base.gj_w); np.add.at(Wg, (base.gj_a, base.gj_b), base.gj_w)
explainable = set()
for i, j in zip(*np.where(negp)):
    if i == j or i not in moa or j not in moa: continue
    mi, mj = moa[i], moa[j]
    if Wi[mi, mj] > 0 or (Wi[mi, :] * ((We + Wg)[:, mj] > 0)).sum() > 0: explainable.add((i, j))
stims = sorted({j for i, j in zip(*np.where(negp)) if i != j and i in moa and j in moa})
if a.exclude:
    excl = set(json.load(open(a.exclude))["train"]); stims = [j for j in stims if ids[j] not in excl]
print(f"stims with inhibitory pairs: {len(stims)}; explainable inhibitory pairs: {len(explainable)}")

dt = a.dt; rec = 4
def build(theta):
    return jaxsim.build_params(apply_theta_class(base, theta) if len(theta) > 8 else apply_theta(base, theta), jnp.float32)
def run(P, dur, I):
    _, Vs = jaxsim.simulate(P, dt, dur, N, I_ext=jnp.asarray(I, jnp.float32), record_every=rec); return np.asarray(Vs)
def pulse(dur, cells, amp, stim_ms):
    I = np.zeros((int(dur / dt), N), np.float32); I[: int(stim_ms / dt), [ix[c] for c in cells]] = amp; return I

def evaluate(theta):
    P = build(theta); r = {"theta": [float(x) for x in theta]}
    Vc = run(P, 2000, np.zeros((8000, N), np.float32)); r["rest_mV"] = float(Vc[-1, ~isM].mean())
    # (a) 안정성
    for name, cells in [("PLM", ["PLML", "PLMR"]), ("AVA", ["AVAL", "AVAR"])]:
        V = run(P, 4000, pulse(4000, cells, 10.0, 4000)); r[f"up_{name}_10pA"] = float((V[-1, ~isM] > -30).mean())
    # (c) 회로 (5 pA, 2 s, 마지막 500 ms)
    sl = slice(int(1500 / dt / rec), None); basec = Vc[sl].mean(0)
    AVB = [ix["AVBL"], ix["AVBR"]]; AVA = [ix["AVAL"], ix["AVAR"]]
    for name, cells in [("PLM", ["PLML", "PLMR"]), ("ALM", ["ALML", "ALMR", "AVM"])]:
        d = run(P, 2000, pulse(2000, cells, 5.0, 2000))[sl].mean(0) - basec
        r[f"circ_{name}_dAVB"] = float(d[AVB].mean()); r[f"circ_{name}_dAVA"] = float(d[AVA].mean())
    r["circ_ok"] = bool(r["circ_PLM_dAVB"] > r["circ_PLM_dAVA"] and r["circ_ALM_dAVA"] > r["circ_ALM_dAVB"])
    # (b) 아틀라스: 억제 쌍이 있는 자극 뉴런
    base2 = Vc.mean(0); X, Y, S, E = [], [], [], []
    for j in stims:
        mj = moa[j]; dV = run(P, 2000, pulse(2000, [ids[j]] if ids[j] in ix else [base.names[mj]], 2.0, 500)).mean(0) - base2
        for i in range(len(ids)):
            if np.isfinite(dff[i, j]) and i != j and i in moa:
                X.append(dV[moa[i]]); Y.append(dff[i, j]); S.append(sig[i, j]); E.append((i, j) in explainable)
    X, Y, S, E = map(np.array, (X, Y, S, E)); neg = S & (Y < 0); pos = S & (Y > 0)
    r["n_pairs"] = int(len(Y)); r["n_inh"] = int(neg.sum()); r["n_inh_expl"] = int((neg & E).sum())
    r["inh_recall"] = float((X[neg] < 0).mean()); r["inh_recall_expl"] = float((X[neg & E] < 0).mean()) if (neg & E).any() else float("nan")
    r["inh_recall_unexpl"] = float((X[neg & ~E] < 0).mean()); r["exc_recall"] = float((X[pos] > 0).mean())
    r["auroc"] = float(roc_auc_score(S, np.abs(X))); r["spearman"] = float(spearmanr(X, Y)[0])
    r["frac_neg_all"] = float((X < 0).mean())      # 모델이 음으로 예측한 비율 (억제 재현율의 기저)
    return r

# 설정: θ8 = Vfit 6개 + [Vth_inh, log δ_inh]; log g_inh 는 c302 억제 전도도 0.29 nS 대비 배율
def theta8(vth_inh, log_g_inh, log_delta_inh):
    t = list(THETA_FIT_C); t[1] = log_g_inh; return t + [vth_inh, log_delta_inh]
if a.theta_class:
    log = json.load(open(a.theta_class)); th = log[a.step]["theta"]; r = evaluate(th); r["source"] = f"{a.theta_class}[{log[a.step]['it']}]"
    json.dump(r, open(a.out, "w"), indent=1)
    print(f"{r['source']} | rest {r['rest_mV']:+.1f} up PLM {r['up_PLM_10pA']:.2f} AVA {r['up_AVA_10pA']:.2f} | inh {r['inh_recall']:.2f} (expl {r['inh_recall_expl']:.2f}, unexpl {r['inh_recall_unexpl']:.2f}) exc {r['exc_recall']:.2f} "
          f"AUROC {r['auroc']:.3f} ρ {r['spearman']:+.3f} neg% {r['frac_neg_all']:.2f} | PLM AVB−AVA {r['circ_PLM_dAVB']-r['circ_PLM_dAVA']:+.1f} ALM AVA−AVB {r['circ_ALM_dAVA']-r['circ_ALM_dAVB']:+.1f} {'OK' if r['circ_ok'] else ''}")
    sys.exit(0)
if a.configs:
    grid = [tuple(float(x) for x in c.split(",")) for c in a.configs.split(";")]
elif a.grid == "quick":
    grid = [(1.4835, THETA_FIT_C[1], 0.7679)]     # = Vfit
    grid += list(itertools.product([-55.0, -50.0, -45.0, -40.0], [np.log(0.1), np.log(0.3), 0.0], [0.0]))
else:
    grid = [(1.4835, THETA_FIT_C[1], 0.7679)] + list(itertools.product([-60.0, -55.0, -50.0, -45.0, -40.0, -30.0], [np.log(0.03), np.log(0.1), np.log(0.3), 0.0, np.log(3.0)], [np.log(0.5), 0.0, np.log(2.0)]))
results = []
if os.path.exists(a.out):
    results = json.load(open(a.out)); done = {tuple(np.round(r["theta"], 4)) for r in results}
else: done = set()
t0 = time.time()
for k, (vth_i, lg_i, ld_i) in enumerate(grid):
    th = theta8(vth_i, lg_i, ld_i)
    if tuple(np.round(th, 4)) in done: continue
    r = evaluate(th); r.update({"vth_inh": vth_i, "g_inh_mult": float(np.exp(lg_i)), "delta_inh_mult": float(np.exp(ld_i))}); results.append(r)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"[{time.time()-t0:5.0f}s] {k+1}/{len(grid)} Vth_inh {vth_i:+5.1f} g_inh x{np.exp(lg_i):.3f} δ x{np.exp(ld_i):.2f} | rest {r['rest_mV']:+.1f} up PLM {r['up_PLM_10pA']:.2f} AVA {r['up_AVA_10pA']:.2f} "
          f"| inh {r['inh_recall']:.2f} (expl {r['inh_recall_expl']:.2f}) exc {r['exc_recall']:.2f} AUROC {r['auroc']:.3f} ρ {r['spearman']:+.3f} neg% {r['frac_neg_all']:.2f} "
          f"| PLM AVB−AVA {r['circ_PLM_dAVB']-r['circ_PLM_dAVA']:+.1f} ALM AVA−AVB {r['circ_ALM_dAVA']-r['circ_ALM_dAVB']:+.1f} {'OK' if r['circ_ok'] else ''}", flush=True)
print("done")
