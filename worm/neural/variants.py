"""모델 변형 (ADR-010). 커넥톰(연결의 존재와 개수)은 절대 바꾸지 않는다. 바꾸는 것은 동역학 가정뿐이다.

V0: c302 C2 기본.
V1 (polarity="cengen"): 뉴런→뉴런 화학 시냅스의 부호를 CeNGEN 수용체 발현 예측(Fenyves 2020 방식, wormneuroatlas)으로 배정.
    예측 없음 → c302 기본 유지. 상충(0) → conflict 정책: "default"(c302 유지) 또는 "split"(가중치 절반씩 흥분+억제).
    부호가 바뀐 시냅스는 c302의 해당 클래스 파라미터(g, k, erev, delta, vth)를 그대로 받는다.
V2 (g_chem_scale, g_gj_scale): 시냅스/갭정션 전도도 전역 스케일.
"""
from __future__ import annotations
import copy, os
import numpy as np
from .connectome import Network

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def _class_params(net: Network):
    """c302 시냅스 클래스별 (g, delta, vth, k, erev) 를 네트워크에서 읽는다."""
    out = {}
    for i, sid in enumerate(net.syn_id):
        if sid not in out:
            out[sid] = (net.syn_g[i], net.syn_delta[i], net.syn_vth[i], net.syn_k[i], net.syn_erev[i])
    return out

def apply_polarity(net: Network, mode: str = "both", conflict: str = "default") -> tuple[Network, dict]:
    d = np.load(os.path.join(DATA, "synapse_sign.npz"))
    ids = list(d["ids"]); sign = d[f"sign_{mode}"]
    aidx = {n: i for i, n in enumerate(ids)}; aidx["AWCL"] = aidx.get("AWCOFF", -1); aidx["AWCR"] = aidx.get("AWCON", -1)
    cp = _class_params(net)
    exc, inh = cp["neuron_to_neuron_exc_syn"], cp["neuron_to_neuron_inh_syn"]
    isM = net.is_muscle()
    cols = {k: list(getattr(net, k)) for k in ["syn_pre", "syn_post", "syn_w", "syn_g", "syn_delta", "syn_vth", "syn_k", "syn_erev", "syn_id"]}
    stats = {"kept": 0, "flipped_to_inh": 0, "flipped_to_exc": 0, "conflict": 0, "nopred": 0, "split": 0}
    n0 = len(net.syn_pre)
    for i in range(n0):
        pre, post = net.syn_pre[i], net.syn_post[i]
        if isM[post] or isM[pre]:
            stats["kept"] += 1; continue
        a, b = aidx.get(net.names[pre], -1), aidx.get(net.names[post], -1)
        s = sign[b, a] if (a >= 0 and b >= 0) else np.nan
        cur_inh = "inh" in net.syn_id[i]
        if np.isnan(s):
            stats["nopred"] += 1; continue
        if s == 0:
            stats["conflict"] += 1
            if conflict == "split":
                stats["split"] += 1
                cols["syn_w"][i] = net.syn_w[i] / 2
                other = inh if not cur_inh else exc
                for k, v in zip(["syn_g", "syn_delta", "syn_vth", "syn_k", "syn_erev"], other): cols[k].append(v)
                cols["syn_pre"].append(pre); cols["syn_post"].append(post); cols["syn_w"].append(net.syn_w[i] / 2)
                cols["syn_id"].append("neuron_to_neuron_inh_syn" if not cur_inh else "neuron_to_neuron_exc_syn")
            continue
        want_inh = s < 0
        if want_inh == cur_inh:
            stats["kept"] += 1; continue
        stats["flipped_to_inh" if want_inh else "flipped_to_exc"] += 1
        p = inh if want_inh else exc
        for k, v in zip(["syn_g", "syn_delta", "syn_vth", "syn_k", "syn_erev"], p): cols[k][i] = v
        cols["syn_id"][i] = "neuron_to_neuron_inh_syn" if want_inh else "neuron_to_neuron_exc_syn"
    out = copy.copy(net)
    for k, v in cols.items():
        setattr(out, k, v if k == "syn_id" else np.asarray(v, dtype=np.int32 if k in ("syn_pre", "syn_post") else np.float64))
    return out, stats

def scale_conductances(net: Network, g_chem: float = 1.0, g_gj: float = 1.0, neuron_only: bool = True) -> Network:
    out = copy.copy(net); isM = net.is_muscle()
    m = ~(isM[net.syn_pre] | isM[net.syn_post]) if neuron_only else np.ones(len(net.syn_pre), bool)
    out.syn_g = np.where(m, net.syn_g * g_chem, net.syn_g)
    mg = ~(isM[net.gj_a] | isM[net.gj_b]) if neuron_only else np.ones(len(net.gj_a), bool)
    out.gj_g = np.where(mg, net.gj_g * g_gj, net.gj_g)
    return out

def make_variant(net: Network, name: str) -> tuple[Network, dict]:
    """이름 규약: V0 | V1[-dominant][-split] | V2-<g_chem>[-gj<g_gj>] | V3-... (V1 옵션 + V2 옵션) | Vfit (= V1-split + THETA_FIT_C)"""
    if name == "Vfit":
        out, info = make_variant(net, "V1-split"); out = apply_theta(out, THETA_FIT_C); info["variant"] = "Vfit"; info["theta"] = THETA_FIT_C; return out, info
    parts = name.split("-"); base = parts[0]; opts = parts[1:]; info = {"variant": name}
    out = net
    if base in ("V1", "V3"):
        mode = "dominant" if "dominant" in opts else "both"; conflict = "split" if "split" in opts else "default"
        out, st = apply_polarity(out, mode, conflict); info.update(st)
    if base in ("V2", "V3"):
        nums = [o for o in opts if o.replace(".", "").isdigit()]; gj = [o[2:] for o in opts if o.startswith("gj")]
        out = scale_conductances(out, g_chem=float(nums[0]) if nums else 1.0, g_gj=float(gj[0]) if gj else 1.0)
        info["g_chem"] = float(nums[0]) if nums else 1.0; info["g_gj"] = float(gj[0]) if gj else 1.0
    vth = [o[3:] for o in opts if o.startswith("vth")]
    if vth:                                    # 뉴런→뉴런 화학 시냅스 반활성 전압 (vthm40 → −40 mV): 휴지 시 긴장성 전달 가정
        v = -float(vth[0][1:]) if vth[0].startswith("m") else float(vth[0])
        out = copy.copy(out); isM = out.is_muscle(); nn = ~(isM[out.syn_pre] | isM[out.syn_post])
        out.syn_vth = np.where(nn, v, out.syn_vth); info["vth_nn"] = v
    leak = [o[4:] for o in opts if o.startswith("leak")]
    if leak:                                   # 뉴런 누설 전도도 배율 (고유 전도도 가정; 근육 제외)
        out = copy.copy(out); out.leak_scale = float(leak[0]); info["leak_scale"] = out.leak_scale
    return out, info


# Phase 1c 적합 결과 (runs/phase1c/fit_C, 학습 32 / 검증 140 자극 뉴런). 기저 변형 V1-split 위에 적용.
THETA_FIT_C = [-3.4394, -6.2547, 0.1307, 0.6922, 1.4835, 0.7679]

THETA_NAMES = ["log_g_exc", "log_g_inh", "log_g_gj", "log_g_leak", "vth_nn", "log_delta_nn"]

def apply_theta(net: Network, theta) -> Network:
    """phase1c_fit.py 의 전역 파라미터 θ(6개)를 Network 수준에서 적용한다 (Params 수준 apply 와 동일한 의미).
    θ = [log g_exc, log g_inh, log g_gj, log g_leak, Vth_nn(mV), log δ_nn]. 뉴런→뉴런 시냅스와 뉴런 누설에만 적용."""
    le, li, lg, ll, vth, ld = [float(x) for x in theta]
    out = copy.copy(net); isM = net.is_muscle()
    nn = ~(isM[net.syn_pre] | isM[net.syn_post]); exc = nn & np.array(["exc" in s for s in net.syn_id]); inh = nn & np.array(["inh" in s for s in net.syn_id])
    out.syn_g = net.syn_g * np.where(exc, np.exp(le), 1.0) * np.where(inh, np.exp(li), 1.0)
    out.syn_vth = np.where(nn, vth, net.syn_vth); out.syn_delta = np.where(nn, net.syn_delta * np.exp(ld), net.syn_delta)
    mg = ~(isM[net.gj_a] | isM[net.gj_b]); out.gj_g = np.where(mg, net.gj_g * np.exp(lg), net.gj_g)
    out.leak_scale = getattr(net, "leak_scale", 1.0) * np.exp(ll)
    return out
