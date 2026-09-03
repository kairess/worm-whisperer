"""c302 파라미터 세트 C2의 JAX 재구현.

모델 (docs/VALIDATION.md 사양):
  C dV/dt = −[g_L(V−E_L) + g_Ks·n(V−E_Ks) + g_Kf·p⁴q(V−E_Kf) + g_Ca·fc(Ca)·e²f(V−E_Ca)] + I_syn + I_gap + I_ext
  게이트 x: dx/dt = (x_inf(V) − x)/τ_x,  x_inf = 1/(1+exp(−(V−mid)/scale))
  fc(Ca) = 1 + (h_inf − 1)·α,  h_inf = 1/(1+exp((ca_half − Ca)/k))
  dCa/dt = ρ'·I_Ca,in − (Ca − Ca_rest)/τ_Ca        (Ca ≥ 0)
  등급형 시냅스: s_inf = 1/(1+exp((Vth − V_pre)/δ)), τ_s = (1−s_inf)/k, ds/dt = (s_inf − s)/τ_s, I = w·g·s·(E_rev − V_post)
  갭정션: I_a = w·g·(V_b − V_a), I_b = w·g·(V_a − V_b)   (NeuroML 연결 하나당 양쪽 모두)

수치: NEURON 고정 스텝과 동일하게 전압은 후진 오일러(전도도는 t 시점), 게이트·s·Ca는 정확 지수 업데이트(cnexp).
단위: mV, ms, pA, nS, pF, mM.
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from .connectome import Network

class Params(NamedTuple):
    C: jnp.ndarray; v_init: jnp.ndarray
    gL: jnp.ndarray; EL: jnp.ndarray
    gKs: jnp.ndarray; EKs: jnp.ndarray
    gKf: jnp.ndarray; EKf: jnp.ndarray
    gCa: jnp.ndarray; ECa: jnp.ndarray
    gate: dict            # name → (tau, mid, scale) 각 (N,)
    ca_alpha: jnp.ndarray; ca_k: jnp.ndarray; ca_half: jnp.ndarray
    ca_tau: jnp.ndarray; ca_rho: jnp.ndarray; ca_rest: jnp.ndarray   # ca_rho: mM/ms per pA
    gKCa: jnp.ndarray; kca_half: jnp.ndarray; EKCa: jnp.ndarray       # 세포 모델 확장 C1: Ca 활성화 K 전류 (기본 0)
    gKw: jnp.ndarray; w_vhalf: jnp.ndarray; w_k: jnp.ndarray; w_tau: jnp.ndarray   # 확장 C2: 느린 K 전류 (SLO-2 계열 근사, 기본 0)
    syn_pre: jnp.ndarray; syn_post: jnp.ndarray; syn_wg: jnp.ndarray
    syn_delta: jnp.ndarray; syn_vth: jnp.ndarray; syn_k: jnp.ndarray; syn_erev: jnp.ndarray
    syn_in: jnp.ndarray       # (N, D_syn) 각 세포로 들어오는 시냅스 인덱스, 패딩은 더미(마지막) 시냅스(전도도 0)
    gj_peer: jnp.ndarray; gj_wg_pad: jnp.ndarray   # (N, D_gj) 갭정션 상대 세포 인덱스와 w·g (패딩 0)
    gj_rowsum: jnp.ndarray    # (N,) Σ w·g
    pulse_mat: jnp.ndarray; pulse_t0: jnp.ndarray; pulse_t1: jnp.ndarray; pulse_amp: jnp.ndarray

class State(NamedTuple):
    V: jnp.ndarray; n: jnp.ndarray; p: jnp.ndarray; q: jnp.ndarray; e: jnp.ndarray; f: jnp.ndarray
    Ca: jnp.ndarray; s: jnp.ndarray; w: jnp.ndarray

UNROLL = 8
GATE_CHANNEL = {"n": "k_slow", "p": "k_fast", "q": "k_fast", "e": "ca_boyle", "f": "ca_boyle"}

def build_params(net: Network, dtype=jnp.float64) -> Params:
    N = net.n
    per = {k: np.zeros(N) for k in ["C", "v_init", "gL", "EL", "gKs", "EKs", "gKf", "EKf", "gCa", "ECa",
                                    "ca_alpha", "ca_k", "ca_half", "ca_tau", "ca_rho", "ca_rest"]}
    gate = {g: (np.zeros(N), np.zeros(N), np.zeros(N)) for g in "npqef"}
    for ti, t in enumerate(net.types):
        m = net.cell_type == ti
        per["C"][m] = t.cap_pF; per["v_init"][m] = t.v_init
        per["ca_tau"][m] = t.ca_decay_ms; per["ca_rest"][m] = t.ca_rest_mM
        # ρ' [mM/ms/pA] = (1e-3 nA/pA / area) × ρ[mol/m/A/s → umol/cm/nA/ms: ×1e-8]
        per["ca_rho"][m] = 1e-3 / t.area_cm2 * t.ca_rho * 1e-8
        for ch, gd, erev in t.chans:
            g_nS = gd * t.area_cm2 * 1e9
            key = {"Leak": "L", "k_slow": "Ks", "k_fast": "Kf", "ca_boyle": "Ca"}[ch.id.replace("_muscle", "")]
            per["g" + key][m] = g_nS; per["E" + key][m] = erev
            for gt in ch.gates:
                assert gt.rate == 1.0
                gate[gt.name][0][m] = gt.tau; gate[gt.name][1][m] = gt.mid; gate[gt.name][2][m] = gt.scale
            if ch.ca_gate:
                per["ca_alpha"][m] = ch.ca_gate.alpha; per["ca_k"][m] = ch.ca_gate.k; per["ca_half"][m] = ch.ca_gate.ca_half
    ls = getattr(net, "leak_scale", 1.0)
    if ls != 1.0:
        per["gL"] = np.where(net.is_muscle(), per["gL"], per["gL"] * ls)
    lsc = getattr(net, "leak_scale_cell", None)          # (N,) 세포별 누설 배율 (Phase 1d 클래스 θ; 이미 leak_scale 을 포함)
    if lsc is not None:
        per["gL"] = np.where(net.is_muscle(), per["gL"], per["gL"] / (ls if ls != 1.0 else 1.0) * np.asarray(lsc))
    A = lambda x: jnp.asarray(x, dtype)
    pulses = net.pulses
    # 패딩된 인접 리스트 (scatter 대신 gather+sum: XLA CPU에서 훨씬 빠름)
    n_syn = len(net.syn_pre)
    syn_in = _padded_lists(net.syn_post, N, fill=n_syn)          # 더미 시냅스 인덱스 = n_syn
    syn_pre = np.append(net.syn_pre, 0); syn_post = np.append(net.syn_post, 0)
    syn_wg = np.append(net.syn_w * net.syn_g, 0.0)
    syn_delta = np.append(net.syn_delta, 1.0); syn_vth = np.append(net.syn_vth, 0.0)
    syn_k = np.append(net.syn_k, 1.0); syn_erev = np.append(net.syn_erev, 0.0)
    ends = np.concatenate([net.gj_a, net.gj_b]); peers = np.concatenate([net.gj_b, net.gj_a])
    wg = np.concatenate([net.gj_w * net.gj_g] * 2)
    if len(ends) == 0:
        gj_peer = np.zeros((N, 1), np.int32); gj_wg_pad = np.zeros((N, 1))
    else:
        gj_lists = _padded_lists(ends, N, fill=-1)
        gj_peer = np.where(gj_lists >= 0, peers[np.maximum(gj_lists, 0)], 0)
        gj_wg_pad = np.where(gj_lists >= 0, wg[np.maximum(gj_lists, 0)], 0.0)
    pulse_mat = np.zeros((N, len(pulses)))
    for j, (c, *_ ) in enumerate(pulses): pulse_mat[int(c), j] = 1.0
    return Params(
        C=A(per["C"]), v_init=A(per["v_init"]), gL=A(per["gL"]), EL=A(per["EL"]), gKs=A(per["gKs"]), EKs=A(per["EKs"]),
        gKf=A(per["gKf"]), EKf=A(per["EKf"]), gCa=A(per["gCa"]), ECa=A(per["ECa"]),
        gate={g: tuple(A(v) for v in vals) for g, vals in gate.items()},
        ca_alpha=A(per["ca_alpha"]), ca_k=A(per["ca_k"]), ca_half=A(per["ca_half"]),
        ca_tau=A(per["ca_tau"]), ca_rho=A(per["ca_rho"]), ca_rest=A(per["ca_rest"]),
        gKCa=A(np.zeros(N)), kca_half=A(np.full(N, 1e-6)), EKCa=A(np.full(N, -70.0)),
        gKw=A(np.zeros(N)), w_vhalf=A(np.full(N, -30.0)), w_k=A(np.full(N, 8.0)), w_tau=A(np.full(N, 300.0)),
        syn_pre=jnp.asarray(syn_pre), syn_post=jnp.asarray(syn_post), syn_wg=A(syn_wg),
        syn_delta=A(syn_delta), syn_vth=A(syn_vth), syn_k=A(syn_k), syn_erev=A(syn_erev),
        syn_in=jnp.asarray(syn_in), gj_peer=jnp.asarray(gj_peer), gj_wg_pad=A(gj_wg_pad), gj_rowsum=A(gj_wg_pad.sum(1)),
        pulse_mat=A(pulse_mat), pulse_t0=A(pulses[:, 1]),
        pulse_t1=A(pulses[:, 1] + pulses[:, 2]), pulse_amp=A(pulses[:, 3]),
    )

def _padded_lists(owner: np.ndarray, N: int, fill: int) -> np.ndarray:
    """owner[i] = 항목 i가 속한 세포. 반환 (N, D): 세포별 항목 인덱스, 빈 칸은 fill."""
    order = np.argsort(owner, kind="stable")
    counts = np.bincount(owner, minlength=N) if len(owner) else np.zeros(N, int); D = max(int(counts.max()), 1)
    out = np.full((N, D), fill, dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for c in range(N):
        out[c, :counts[c]] = order[starts[c]:starts[c] + counts[c]]
    return out

def _xinf(V, g):
    tau, mid, scale = g
    return 1.0 / (1.0 + jnp.exp(-(V - mid) / scale))

def init_state(P: Params) -> State:
    V = P.v_init
    n, p, q, e, f = (_xinf(V, P.gate[g]) for g in "npqef")
    Ca = P.ca_rest * 0.0            # NEURON initialConcentration = 0
    s_inf = 1.0 / (1.0 + jnp.exp((P.syn_vth - V[P.syn_pre]) / P.syn_delta))
    w = 1.0 / (1.0 + jnp.exp(-(V - P.w_vhalf) / P.w_k))
    return State(V, n, p, q, e, f, Ca, s_inf, w)

def _conductances(P: Params, S: State):
    h_inf = 1.0 / (1.0 + jnp.exp((P.ca_half - S.Ca) / P.ca_k))
    fc = 1.0 + (h_inf - 1.0) * P.ca_alpha
    gKs = P.gKs * S.n
    gKf = P.gKf * S.p ** 4 * S.q
    gCa = P.gCa * fc * S.e ** 2 * S.f
    gKCa = P.gKCa * S.Ca ** 2 / (S.Ca ** 2 + P.kca_half ** 2)      # Hill n=2 (SLO-2 형, Gao 2018 계열 가정)
    gKCa = gKCa + P.gKw * S.w                                         # 느린 전압 의존 K (C2)
    return gKs, gKf, gCa, gKCa

def make_step(P: Params, dt: float, n_cells: int):
    N = n_cells
    def step(S: State, t_and_iext):
        t, I_ext_block = t_and_iext
        V = S.V
        gKs, gKf, gCa, gKCa = _conductances(P, S)
        # 화학 시냅스 (전도도는 t 시점의 s), 세포별 gather 후 합산
        Gs = P.syn_wg * S.s
        Gs_in = Gs[P.syn_in]
        Gsyn = Gs_in.sum(1)
        GsynE = (Gs_in * P.syn_erev[P.syn_in]).sum(1)
        # 갭정션: 양방향 (테이블에 이미 양쪽 포함)
        Ggj = P.gj_rowsum
        GgjV = (P.gj_wg_pad * V[P.gj_peer]).sum(1)
        # 펄스 입력 (t ≥ t0 이고 t < t1)
        on = (t >= P.pulse_t0) & (t < P.pulse_t1)
        I_ext = P.pulse_mat @ jnp.where(on, P.pulse_amp, 0.0) + I_ext_block
        Gtot = P.gL + gKs + gKf + gCa + gKCa + Gsyn + Ggj
        GE = P.gL * P.EL + gKs * P.EKs + gKf * P.EKf + gCa * P.ECa + gKCa * P.EKCa + GsynE + GgjV + I_ext
        a = dt / P.C
        Vn = (V + a * GE) / (1.0 + a * Gtot)                     # 후진 오일러
        # 게이트 (정확 지수)
        def gate(x, g):
            tau = g[0]
            return x + (_xinf(Vn, g) - x) * (1.0 - jnp.exp(-dt / tau))
        n, p, q, e, f = (gate(getattr(S, k), P.gate[k]) for k in "npqef")
        # 칼슘: 선형 ODE 정확해. 소스는 t 시점 전도도 × 새 전압
        I_ca_in = gCa * (P.ECa - Vn)
        Ca_inf = P.ca_rest + P.ca_tau * P.ca_rho * I_ca_in
        Ca = Ca_inf + (S.Ca - Ca_inf) * jnp.exp(-dt / P.ca_tau)
        Ca = jnp.maximum(Ca, 0.0)
        # 시냅스 s (전시냅스 새 전압)
        s_inf = 1.0 / (1.0 + jnp.exp((P.syn_vth - Vn[P.syn_pre]) / P.syn_delta))
        one_m = 1.0 - s_inf
        tau_s = one_m / P.syn_k
        s_exp = S.s + (s_inf - S.s) * (1.0 - jnp.exp(-dt / jnp.maximum(tau_s, 1e-12)))
        s = jnp.where(one_m < 1e-4, s_inf, s_exp)
        w_inf = 1.0 / (1.0 + jnp.exp(-(Vn - P.w_vhalf) / P.w_k))
        w = S.w + (w_inf - S.w) * (1.0 - jnp.exp(-dt / P.w_tau))
        return State(Vn, n, p, q, e, f, Ca, s, w), Vn
    return step

_COMPILED = {}

def compiled_run(dt: float, n_cells: int, record_every: int = 1):
    """(P, S, ts, I_ext) → (S, Vs) 를 jit 컴파일해 캐시. P는 인자로 받아 파라미터를 바꿔도 재컴파일하지 않는다."""
    key = (dt, n_cells, record_every)
    if key not in _COMPILED:
        def run(P, S, ts, I_ext):
            S, Vs = lax.scan(make_step(P, dt, n_cells), S, (ts, I_ext), unroll=UNROLL)
            return S, Vs[::record_every]
        _COMPILED[key] = jax.jit(run)
    return _COMPILED[key]

def simulate(P: Params, dt: float, duration: float, n_cells: int, I_ext=None, S0: State | None = None,
             record_every: int = 1):
    """duration ms 동안 시뮬레이션. I_ext: (steps, N) 또는 None. 반환 (최종 상태, V 기록 (steps/record_every, N))."""
    steps = int(round(duration / dt))
    S = init_state(P) if S0 is None else S0
    ts = jnp.arange(steps, dtype=P.C.dtype) * dt
    if I_ext is None:
        I_ext = jnp.zeros((steps, n_cells), P.C.dtype)
    return compiled_run(dt, n_cells, record_every)(P, S, ts, I_ext)

class WormSim:
    """대화형 사용: 50 ms 블록 단위로 외부 전류를 바꾸며 진행."""
    def __init__(self, net: Network, dt=0.05, dtype=jnp.float64):
        self.net, self.dt, self.N = net, dt, net.n
        self.P = build_params(net, dtype)
        self.S = init_state(self.P)
        self.t = 0.0
        self._run = compiled_run(dt, self.N)
        self.index = {n: i for i, n in enumerate(net.names)}
    def run(self, block_ms: float, I_ext: dict[str, float] | None = None):
        steps = int(round(block_ms / self.dt))
        ts = self.t + jnp.arange(steps, dtype=self.P.C.dtype) * self.dt
        I = np.zeros(self.N)
        for name, amp in (I_ext or {}).items():
            I[self.index[name]] += amp
        I = jnp.broadcast_to(jnp.asarray(I, self.P.C.dtype), (steps, self.N))
        self.S, Vs = self._run(self.P, self.S, ts, I)
        self.t += block_ms
        return np.asarray(Vs)
