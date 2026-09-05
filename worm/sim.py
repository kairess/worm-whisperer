"""신경 모델(JAX c302 C2 변형) + 2D 신체 모델 결합 시뮬레이터.

블록(기본 50 ms)마다: 신경 모델 진행 → 근육 [Ca] → 활성 A = Ca/(Ca+K_half) → 행별 등/배 → 선호 곡률 → 신체 진행 → 곡률.
고유수용(옵션): 관절 곡률 → B형 운동뉴런 전류 (Wen 2012 방식; 커넥톰 변경 아님, 감각 입력 추가).
"""
from __future__ import annotations
import re, numpy as np, jax.numpy as jnp
from .neural.connectome import load_network, Network
from .neural.variants import make_variant, apply_theta, apply_theta_class
from .neural import jaxsim
from .body.rod2d import Rod2D
from .body.muscle_map import muscle_rows, activation_from_ca, preferred_curvature
from .body.boyle_motor import BoyleMotor
from .body import motor_block

class Worm:
    def __init__(self, nml: str, variant="V0", theta=None, dt_neural=0.25, block_ms=50.0, K_half=2e-7, kappa_max=10.0,
                 proprio_gain=0.0, proprio_offset=1, proprio_sign=+1, Cn=40.0, dtype=jnp.float32,
                 motor="c302", boyle_kw=None, gate_thresh=3.0, gate_slope=1.0):
        """motor: "c302" (근육 세포 Ca → 곡률) 또는 "boyle" (Vfit 의 AVB/AVA 전압으로 Boyle 2012 운동층을 게이트)."""
        net = load_network(nml); net.pulses = np.zeros((0, 4)); net, self.vinfo = make_variant(net, variant)
        if theta is not None: net = apply_theta_class(net, theta) if len(theta) > 8 else apply_theta(net, theta)   # θ6/θ8 전역 또는 θ63 클래스별 (ADR-015)
        self.net: Network = net; self.neural = jaxsim.WormSim(net, dt=dt_neural, dtype=dtype)
        self.body = Rod2D(Cn=Cn); self.x = self.body.initial(); self.block_ms = block_ms   # dt 10 µs (20 µs 는 굽힘 항 때문에 발산)
        self.D_rows, self.V_rows = muscle_rows(net.names); self.K_half, self.kappa_max = K_half, kappa_max
        self.proprio_gain, self.proprio_offset, self.proprio_sign = proprio_gain, proprio_offset, proprio_sign
        self.DB = [net.index(f"DB{i}") for i in range(1, 8)]; self.VB = [net.index(f"VB{i}") for i in range(1, 12)]
        # 운동뉴런 위치 → 관절 인덱스 (앞→뒤 순서로 균등 배치, 근사)
        self.DB_joint = np.linspace(0, self.body.n - 2, len(self.DB)).astype(int); self.VB_joint = np.linspace(0, self.body.n - 2, len(self.VB)).astype(int)
        self.kappa = np.zeros(self.body.n - 1); self.t = 0.0; self.log = []
        self.motor = motor; kw = dict(G_SR=2.0, sr_span_frac=0.25, I_avb_D=0.5, I_avb_V=0.5); kw.update(boyle_kw or {})
        self.fwd = BoyleMotor(direction=+1, **kw); self.bwd = BoyleMotor(direction=-1, **dict(kw, sr_span_frac=0.5)); self.kappa_max_boyle = 6.0
        self.gate_thresh, self.gate_slope = gate_thresh, gate_slope
        self.AVB = [net.index("AVBL"), net.index("AVBR")]; self.AVA = [net.index("AVAL"), net.index("AVAR")]
        self.SMDD = [net.index("SMDDL"), net.index("SMDDR")]; self.SMDV = [net.index("SMDVL"), net.index("SMDVR")]; self.RIS = net.index("RIS")
        self.RIV = [net.index("RIVL"), net.index("RIVR")]
        self.V_rest = None; self.motor_dt = 5e-3   # 운동층 갱신 5 ms (근육 τ 100 ms 대비 충분)
        self.k_turn = 0.03         # 조향: 앞쪽 head_rows 행의 선호 곡률 편향 = k_turn × (ΔV_SMDD − ΔV_SMDV) [1/mm per mV] (Gray et al. 2005 SMD 머리 조향)
        self.k_steer = 0.0         # (구) 문턱 편향 방식, 사용 안 함
        self.head_rows = 12        # 앞쪽 절반 (SMD 돌기는 준측삭을 따라 몸 중간까지 뻗음). 6 행이면 수용기 창(6 행 이동 창)과 어긋나 파동 주파수가 1 Hz 로 올랐다 (2026-09-03, runs/phase3/turn_freq_scan.txt)
        # 오메가턴 (ADR-016 후보): RIV 활성(ΔV > riv_thresh) 이 시작되면 깊은 배쪽 굽힘 펄스가 머리에서 꼬리로 omega_T 초에 걸쳐 이동한다
        # (Gray 2005: RIV/SMDV 가 오메가턴의 배쪽 깊은 굽힘; Broekmans 2016: 오메가 = 머리→꼬리로 전파하는 깊은 배쪽 굽힘). 진폭 = −k_omega × (ΔV_RIV − riv_thresh).
        # omega_T = 0 이면 정적 편향(앞쪽 omega_rows 행). k_omega = 0 이면 비활성 (기본).
        # 채택값 (runs/phase3/omega_scan_*.txt): 문턱 30 mV 는 직접 자극(ΔV_RIV ≈ 55) 만 통과시키고 갭정션 누설(SMD 자극 시 ≈ 20) 은 무시. 유턴 구간 2 s 에 +162° 재정향.
        self.k_omega = 0.36; self.riv_thresh = 30.0; self.omega_T = 2.5; self.omega_sigma = 4.0; self.omega_rows = 12; self._omega_t0 = None
        self.omega_smd_dir = False; self.omega_dir_thresh = 10.0   # ADR-017 후보: True 면 펄스 방향 = SMDD−SMDV 부호 (배치 환경과 동일 규칙)
        self.ris_gate = True       # 확장 X: RIS 활성(ΔV > gate_thresh) 시 운동층 정지 (Turek et al. 2016). 커넥톰 외 가정이므로 플래그
        self.fast_motor = True     # True: 운동층+막대를 JAX 한 블록으로 (motor_block.py, numpy 경로와 동일 결과·3배 빠름). False: numpy 경로 (검증용)
        self._mb = None; self._mb_state = None
        self.override = None       # Worm Gym 상한 대조군용: {"g_f","g_b","head_bias","omega_bias"} 를 주면 신경 판독 대신 그 값으로 운동층을 구동 (커넥톰 우회, 원칙 위반 — 자 역할만)
        self.last = {}

    def muscle_activation(self):
        Ca = np.asarray(self.neural.S.Ca)
        A = activation_from_ca(Ca, self.K_half)
        A_D = np.array([A[r].mean() if r else 0.0 for r in self.D_rows]); A_V = np.array([A[r].mean() if r else 0.0 for r in self.V_rows])
        return A_D, A_V

    def proprio_currents(self):
        """앞쪽 관절의 굽힘을 같은 방향으로 전파: 등쪽 굽힘(κ>0) → DB, 배쪽 → VB. 관절 j−1 의 곡률을 사용."""
        I = {}
        if self.proprio_gain == 0: return I
        d, sg = self.proprio_offset, self.proprio_sign          # sign +1: 같은 쪽 강화(Wen 2012), −1: 반대쪽 활성화(Boyle 2012 신장 수용)
        for cell, j in zip(self.DB, self.DB_joint):
            k = sg * self.kappa[max(j - d, 0)]; I[self.net.names[cell]] = self.proprio_gain * max(k, 0.0)
        for cell, j in zip(self.VB, self.VB_joint):
            k = sg * self.kappa[max(j - d, 0)]; I[self.net.names[cell]] = self.proprio_gain * max(-k, 0.0)
        return I

    def step(self, I_ext: dict | None = None):
        I = dict(I_ext or {})
        for k, v in self.proprio_currents().items(): I[k] = I.get(k, 0.0) + v
        self.neural.run(self.block_ms, I)
        if self.motor == "c302":
            A_D, A_V = self.muscle_activation(); kappa0 = preferred_curvature(A_D, A_V, self.kappa_max)
            steps = int(round(self.block_ms / 1000 / self.body.dt))
            self.x = self.body.run(self.x, np.broadcast_to(kappa0, (steps, len(kappa0)))); gates = (np.nan, np.nan)
        else:
            V = np.asarray(self.neural.S.V)
            if self.V_rest is None:            # 휴지 기준: 무자극 예열 2 s 후의 막전위 (초기 과도 제외)
                warm = jaxsim.WormSim(self.net, dt=self.neural.dt, dtype=self.neural.P.C.dtype)
                for _ in range(40): warm.run(50.0, {})
                self.V_rest = np.asarray(warm.S.V).copy()
            dV = V - self.V_rest
            dB = dV[self.AVB].mean(); dA = dV[self.AVA].mean(); dSMD = dV[self.SMDD].mean() - dV[self.SMDV].mean(); dRIS = dV[self.RIS]
            # 상호배타 게이트: 더 큰 쪽만, 문턱 이상일 때만 (Boyle: 명령 전류가 회로를 켜고 끔)
            g_f = 1.0 if (dB > self.gate_thresh and dB > dA) else 0.0; g_b = 1.0 if (dA > self.gate_thresh and dA >= dB) else 0.0
            dSMDmax = max(dV[self.SMDD].mean(), dV[self.SMDV].mean())
            quiescent = self.ris_gate and dRIS > self.gate_thresh and dRIS > 1.5 * max(dB, dA, dSMDmax)   # RIS 가 다른 명령/조향 뉴런보다 뚜렷이 더 활성일 때만
            if quiescent: g_f = g_b = 0.0
            gates = (g_f, g_b); head_bias = self.k_turn * dSMD; steer = self.k_steer * dSMD
            dRIV = dV[self.RIV].mean(); omega_bias = -self.k_omega * max(dRIV - self.riv_thresh, 0.0)   # 배쪽(κ<0) 깊은 굽힘 진폭
            if self.omega_smd_dir and dSMD > self.omega_dir_thresh: omega_bias = -omega_bias             # 등쪽 굽힘
            if omega_bias != 0 and self._omega_t0 is None: self._omega_t0 = self.t
            elif omega_bias == 0: self._omega_t0 = None
            if self.override is not None:
                o = self.override; g_f, g_b = float(o.get("g_f", g_f)), float(o.get("g_b", g_b)); head_bias = float(o.get("head_bias", head_bias)); omega_bias = float(o.get("omega_bias", omega_bias))
                if omega_bias < 0 and self._omega_t0 is None: self._omega_t0 = self.t
                elif omega_bias == 0: self._omega_t0 = None
                gates = (g_f, g_b)
            self.last = {"dAVB": float(dB), "dAVA": float(dA), "dSMD": float(dSMD), "dRIS": float(dRIS), "dRIV": float(dRIV), "quiescent": bool(quiescent),
                         "head_bias": float(head_bias), "omega_bias": float(omega_bias), "steer": float(steer)}
            steps = int(round(self.motor_dt / self.body.dt)); n_blocks = int(round(self.block_ms / 1000 / self.motor_dt))
            koff = np.zeros(24); koff[: self.head_rows] += head_bias                                         # 조향(머리) 편향 (근육 행 24)
            if omega_bias != 0:
                if self.omega_T > 0:                                                                          # 이동 펄스: 중심이 머리(−σ) → 꼬리(24+σ)
                    c = -self.omega_sigma + (24 + 2 * self.omega_sigma) * ((self.t - self._omega_t0) / 1000) / self.omega_T
                    koff += omega_bias * np.exp(-((np.arange(24) - c) / self.omega_sigma) ** 2)
                else: koff[: self.omega_rows] += omega_bias                                                    # 정적 편향
            koff_j = 0.5 * (koff[1:] + koff[:-1])                                                             # 관절 23 (인접 행 평균)
            if self.fast_motor:
                if self._mb is None:
                    self._mb = motor_block.make_motor_block(self.body, self.fwd, self.bwd, self.motor_dt, self.block_ms, self.kappa_max_boyle)
                if self._mb_state is None: self._mb_state = motor_block.init_state(self.x, self.fwd, self.bwd, self.body.dtype)
                st = (jnp.asarray(self.x, self.body.dtype),) + tuple(self._mb_state[1:])
                st, A_D, A_V, kap = self._mb(st, g_f, g_b, steer, jnp.asarray(koff, self.body.dtype), jnp.asarray(koff_j, self.body.dtype))
                self._mb_state = st; self.x = st[0]; A_D, A_V = np.asarray(A_D), np.asarray(A_V); n_blocks = 0
            for _ in range(n_blocks):
                kr = np.concatenate([[self.kappa[0]], 0.5 * (self.kappa[1:] + self.kappa[:-1]), [self.kappa[-1]]])
                fD, fV = self.fwd.step(kr, self.motor_dt, g_f, steer, kappa_offset=koff); bD, bV = self.bwd.step(kr, self.motor_dt, g_b, steer, kappa_offset=koff)
                A_D = np.clip(fD + bD, 0, 1); A_V = np.clip(fV + bV, 0, 1)
                kappa0 = preferred_curvature(A_D, A_V, self.kappa_max_boyle)
                kappa0 = kappa0 + koff_j                                             # 선호 곡률에 같은 편향 (수용기 기준도 함께 이동, ADR-013)
                self.x = self.body.run(self.x, np.broadcast_to(kappa0, (steps, len(kappa0)))); self.kappa = np.asarray(self.body.curvature(self.x))
        self.kappa = np.asarray(self.body.curvature(self.x)); self.t += self.block_ms
        rec = {"t": self.t, "x": np.asarray(self.x).copy(), "kappa": self.kappa.copy(), "A_D": A_D, "A_V": A_V, "gates": gates, "readout": dict(self.last), "I_ext": dict(I)}
        self.log.append(rec); return rec

    def run(self, seconds: float, I_ext=None):
        for _ in range(int(round(seconds * 1000 / self.block_ms))): self.step(I_ext)
        return self.log

    def run_schedule(self, schedule, seconds: float):
        """schedule: [(t0_s, t1_s, {neuron: pA}), ...] 시간창별 자극. 겹치면 합산."""
        for _ in range(int(round(seconds * 1000 / self.block_ms))):
            t = self.t / 1000; I = {}
            for t0, t1, stim in schedule:
                if t0 <= t < t1:
                    for k, v in stim.items(): I[k] = I.get(k, 0.0) + v
            self.step(I)
        return self.log

def behavior_descriptors(log, block_ms=50.0, skip_s=1.0):
    """행동 기술자 (m=8): 전진 속도, 후진 비율, 활동량(|v|), 회전율(deg/s, 체축 기준 좌+), 순 회전(deg), 곡률 진폭, 파동 주파수, 정지 비율."""
    xs = np.stack([r["x"] for r in log]); n0 = int(skip_s * 1000 / block_ms); xs = xs[n0:]; dt = block_ms / 1000
    c = xs.mean(1); v = np.gradient(c, dt, axis=0)
    hd = xs[:, 0] - xs[:, -1]; hd /= np.linalg.norm(hd, axis=1, keepdims=True) + 1e-12
    v_ax = np.sum(v * hd, 1); speed = np.linalg.norm(v, axis=1)
    ang = np.unwrap(np.arctan2(hd[:, 1], hd[:, 0])); turn_rate = np.degrees(np.gradient(ang, dt))
    kap = np.stack([r["kappa"] for r in log[n0:]]); kmid = kap[:, kap.shape[1] // 2]
    f = np.fft.rfftfreq(len(kmid), dt); P = np.abs(np.fft.rfft(kmid - kmid.mean())) ** 2; freq = float(f[P[1:].argmax() + 1]) if len(kmid) > 8 else 0.0
    gates = np.array([r.get("gates", (0, 0)) for r in log[n0:]], float)
    return {"v_forward": float(v_ax.mean()), "frac_backward": float((v_ax < -0.02).mean()), "activity": float(speed.mean()),
            "turn_rate": float(turn_rate.mean()), "net_turn_deg": float(np.degrees(ang[-1] - ang[0])), "kappa_amp": float(kap.std()),
            "freq": freq, "frac_still": float(((gates.sum(1) == 0)).mean())}

def kinematics_from_log(log, block_ms=50.0, skip_s=2.0):
    from .body.rod2d import kinematics, Rod2D
    xs = np.stack([r["x"] for r in log]); n0 = int(skip_s * 1000 / block_ms)
    k = kinematics(Rod2D(), xs[n0:], block_ms / 1000)
    AD = np.stack([r["A_D"] for r in log[n0:]]); AV = np.stack([r["A_V"] for r in log[n0:]]); diff = AD - AV
    # 진행파 지표: 행 6 과 행 16 의 (A_D − A_V) 시계열 상호상관 최대 지연(s) 과 시간 변동 표준편차
    a, b = diff[:, 5] - diff[:, 5].mean(), diff[:, 15] - diff[:, 15].mean()
    lag = (np.argmax(np.correlate(b, a, "full")) - (len(a) - 1)) * block_ms / 1000 if a.std() > 1e-6 and b.std() > 1e-6 else np.nan
    k.update({"dv_std_time": float(diff.std(0).mean()), "dv_mean_abs": float(np.abs(diff.mean(0)).mean()), "lag_row6_16_s": float(lag)})
    return k
