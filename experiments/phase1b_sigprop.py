"""Phase 1b: 모델 변형을 Randi et al. 2023 신호전파 아틀라스와 비교한다.

프로토콜: 아틀라스에서 자극된 뉴런 j 마다, 모델에서 j 에 정전류(amp pA, stim_ms)를 주고 모든 뉴런 i 의 ΔV_i
(자극 구간+여파 window 평균 − 무자극 대조군)를 계산한다. 아틀라스의 dFF[i,j](평균 ΔF/F)와 비교.
지표: (a) 측정된 쌍 전체에서 Pearson/Spearman 상관, (b) 유의 쌍(q<0.05)의 부호 일치율, (c) |ΔV| 로 유의/비유의 쌍 분류 AUROC.
대조: 해부학적 직접 연결 강도(화학 개수 + 갭정션 개수, j→i)를 같은 지표로 평가.
사용: uv run python experiments/phase1b_sigprop.py <net.nml> --variant V1 [--amp 2] [--limit 20]
"""
import os, sys, json, time, argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
sys.path.insert(0, os.getcwd())
from worm.neural.connectome import load_network
from worm.neural.variants import make_variant
from worm.neural import jaxsim

ap = argparse.ArgumentParser(); ap.add_argument("nml"); ap.add_argument("--variant", default="V0"); ap.add_argument("--tag", default="")
ap.add_argument("--amp", type=float, default=2.0); ap.add_argument("--stim_ms", type=float, default=1000); ap.add_argument("--dur", type=float, default=2000)
ap.add_argument("--dt", type=float, default=0.25); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--min_measured", type=int, default=30)
ap.add_argument("--win", default=None, help="ΔV 평균 창 'start,end' ms (기본: 전체 dur)")
ap.add_argument("--theta", default=None, help="phase1c 전역 파라미터 6개 (쉼표 구분) 또는 log.json 경로(마지막 스텝)")
ap.add_argument("--theta_class", default=None, help="phase1d 클래스별 θ(63): log.json 경로(마지막 스텝) 또는 쉼표 구분")
ap.add_argument("--exclude", default=None, help="split.json 경로: 그 안의 train 자극 뉴런을 평가에서 제외")
a = ap.parse_args()

net0 = load_network(a.nml); net0.pulses = np.zeros((0, 4)); net, vinfo = make_variant(net0, a.variant); print("variant:", vinfo)
if a.theta:
    from worm.neural.variants import apply_theta
    th = json.load(open(a.theta))[-1]["theta"] if a.theta.endswith(".json") else [float(x) for x in a.theta.split(",")]
    net = apply_theta(net, th); vinfo["theta"] = th; print("theta:", np.round(th, 3).tolist())
if a.theta_class:
    from worm.neural.variants import apply_theta_class
    th = json.load(open(a.theta_class))[-1]["theta"] if a.theta_class.endswith(".json") else [float(x) for x in a.theta_class.split(",")]
    net = apply_theta_class(net, th); vinfo["theta_class"] = th; print("theta_class: 63 params from", a.theta_class)
P = jaxsim.build_params(net, jnp.float32); N = net.n; isM = net.is_muscle()
atl = np.load("worm/data/randi2023_sigprop.npz"); ids = [str(x) for x in atl["ids"]]; dff, q = atl["dff"], atl["q"]
aidx = {n: i for i, n in enumerate(ids)}; aidx["AWCL"] = aidx["AWCOFF"]; aidx["AWCR"] = aidx["AWCON"]
model_of_atlas = {i: net.index(n) for n, i in aidx.items() if n in net.names}      # atlas idx → model idx
atlas_of_model = {m: i for i, m in model_of_atlas.items()}
meas = np.isfinite(dff).sum(0)
stims = [j for j in np.argsort(-meas) if meas[j] >= a.min_measured and j in model_of_atlas]
if a.exclude:
    excl = set(json.load(open(a.exclude))["train"]); stims = [j for j in stims if ids[j] not in excl]; print(f"excluding {len(excl)} train stims")
if a.limit: stims = stims[:a.limit]
print(f"stimulated neurons: {len(stims)}; amp {a.amp} pA, stim {a.stim_ms} ms, window {a.dur} ms, dt {a.dt}")

steps = int(round(a.dur / a.dt)); rec = 4
def run(I):
    _, Vs = jaxsim.simulate(P, a.dt, a.dur, N, I_ext=jnp.asarray(I, jnp.float32), record_every=rec); return np.asarray(Vs)
w0, w1 = (0, a.dur) if a.win is None else [float(x) for x in a.win.split(",")]
sl = slice(int(w0 / a.dt / rec), int(w1 / a.dt / rec))
Vc = run(np.zeros((steps, N), np.float32)); base = Vc[sl].mean(0)
# 해부학적 직접 연결 강도 (j→i): 화학 시냅스 개수 + 갭정션 개수 (모델 인덱스)
W = np.zeros((N, N)); np.add.at(W, (net.syn_post, net.syn_pre), net.syn_w); np.add.at(W, (net.gj_b, net.gj_a), net.gj_w); np.add.at(W, (net.gj_a, net.gj_b), net.gj_w)

rows = []; t0 = time.time(); pooled = {"dv": [], "dff": [], "sig": [], "anat": [], "stim": [], "resp": []}
for k, j in enumerate(stims):
    mj = model_of_atlas[j]
    I = np.zeros((steps, N), np.float32); I[: int(a.stim_ms / a.dt), mj] = a.amp
    V = run(I); dV = V[sl].mean(0) - base
    up = float((V[-1, ~isM] > -30).mean())
    ii = [i for i in range(len(ids)) if np.isfinite(dff[i, j]) and i != j and i in model_of_atlas]
    mi = np.array([model_of_atlas[i] for i in ii]); x = dV[mi]; y = dff[ii, j]; sig = q[ii, j] < 0.05; anat = W[mi, mj]
    pooled["dv"].append(x); pooled["dff"].append(y); pooled["sig"].append(sig); pooled["anat"].append(anat)
    pooled["stim"].append(np.full(len(ii), j)); pooled["resp"].append(np.array(ii))
    from scipy.stats import pearsonr, spearmanr
    r = pearsonr(x, y)[0] if x.std() > 0 else np.nan; rho = spearmanr(x, y)[0] if x.std() > 0 else np.nan
    rows.append({"stim": ids[j], "n": len(ii), "n_sig": int(sig.sum()), "self_dV": float(dV[mj]), "pearson": r, "spearman": rho, "frac_up": up})
    if k % 10 == 0: print(f"[{time.time()-t0:5.0f}s] {k+1}/{len(stims)} {ids[j]:6s} n={len(ii):3d} sig={int(sig.sum()):3d} selfΔV={dV[mj]:+6.1f} r={r:+.2f} ρ={rho:+.2f} up={up:.2f}")

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
X = np.concatenate(pooled["dv"]); Y = np.concatenate(pooled["dff"]); S = np.concatenate(pooled["sig"]); An = np.concatenate(pooled["anat"])
def metrics(pred, name):
    r = pearsonr(pred, Y)[0]; rho = spearmanr(pred, Y)[0]
    auc = roc_auc_score(S, np.abs(pred)) if S.any() else np.nan
    m = S & (np.abs(pred) > 1e-3 * np.abs(pred).max()); sign_acc = float((np.sign(pred[m]) == np.sign(Y[m])).mean()) if m.any() else np.nan
    # 유의 쌍만의 상관
    r_sig = pearsonr(pred[S], Y[S])[0] if S.sum() > 2 else np.nan
    return {"predictor": name, "pearson_all": r, "spearman_all": rho, "pearson_sig": r_sig, "auroc_sig": auc, "sign_acc_sig": sign_acc, "n_pairs": int(len(Y)), "n_sig": int(S.sum())}
res = {"variant": a.variant, "info": vinfo, "amp": a.amp, "stim_ms": a.stim_ms, "dur": a.dur, "dt": a.dt, "n_stim": len(stims),
       "mean_frac_up": float(np.mean([r["frac_up"] for r in rows])), "per_stim": rows,
       "pooled": [metrics(X, "model ΔV"), metrics(An, "anatomy (direct count)"), metrics(np.random.default_rng(0).permutation(X), "shuffled model")]}
os.makedirs("runs/phase1b", exist_ok=True); tag = f"{a.tag+'_' if a.tag else ''}{a.variant}_amp{a.amp:g}_stim{a.stim_ms:g}" + (f"_win{w0:g}-{w1:g}" if a.win else "") + ("_test" if a.exclude else "")
json.dump(res, open(f"runs/phase1b/sigprop_{tag}.json", "w"), indent=1, default=float)
np.savez_compressed(f"runs/phase1b/sigprop_{tag}.npz", dv=X, dff=Y, sig=S, anat=An, stim=np.concatenate(pooled["stim"]), resp=np.concatenate(pooled["resp"]), ids=np.array(ids))
print(f"\n== {tag}: {len(stims)} stim neurons, {len(Y)} pairs ({int(S.sum())} significant), mean frac up-state {res['mean_frac_up']:.2f}")
print(f"{'predictor':24s} {'r_all':>7s} {'rho_all':>8s} {'r_sig':>7s} {'AUROC':>6s} {'sign%':>6s}")
for m in res["pooled"]:
    print(f"{m['predictor']:24s} {m['pearson_all']:+7.3f} {m['spearman_all']:+8.3f} {m['pearson_sig']:+7.3f} {m['auroc_sig']:6.3f} {m['sign_acc_sig']:6.2f}")
print(f"per-stim mean Pearson: {np.nanmean([r['pearson'] for r in rows]):+.3f}, Spearman {np.nanmean([r['spearman'] for r in rows]):+.3f}")
