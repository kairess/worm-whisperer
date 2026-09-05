"""Worm Gym 배치 시뮬레이터: 환경 B 개를 GPU 에서 동시에 굴리는 JAX 구현 (docs/PLAN_WORMGYM.md 1절, 2단계).

worm/sim.py 의 Worm(Boyle 운동층 경로) + worm/env/gym.py 의 WormChemEnv 와 같은 계산을 lax.scan / vmap 으로 다시 쓴 것.
- 신경층: jaxsim.make_step 을 그대로 vmap (파라미터 P 공유, 상태·전류만 배치).
- 판독·게이트·조향·오메가 펄스: Worm.step 과 같은 규칙 (ADR-012/013/016), 오메가 시작 시각은 상태로 들고 다닌다.
- 운동층+막대: body/motor_block.py 를 vmap.
- 정책(MLP)도 롤아웃 안에서 평가하므로 진화 전략의 개체군 전체를 rollout 한 번으로 평가한다.
동등성: tests/test_batch_env.py 에서 WormChemEnv 한 에피소드와 비교.
"""
from __future__ import annotations
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from ..neural.connectome import load_network
from ..neural.variants import make_variant
from ..neural import jaxsim
from ..body.rod2d import Rod2D
from ..body.boyle_motor import BoyleMotor
from ..body import motor_block
from ..llm.protocols import CHANNELS
from .gym import NML

class BatchWormEnv:
    def __init__(self, channels=("AVB", "AVA", "SMDD", "SMDV", "RIV"), mode="whitelist", dt_action=0.5, episode_s=40.0, n_hist=5, amp_max=6.0,
                 src_dist=2.5, sigma=2.0, reach_r=0.5, variant="Vfit", nml=NML, hidden=16, dt_neural=0.25, block_ms=50.0, dtype=jnp.float32,
                 living_cost=0.05, terminal_dist_w=2.0, dwell=False, move_cost=0.0,
                 walls=None, touch_obs=False, starts=None, goals=None, k_w=5e3, wall_r=0.06, no_odor=False, omega_smd_dir=False, omega_dir_thresh=10.0, zero_touch=False, proprio_obs=False, lateral_obs=False, lateral_d=0.1, omega_lock=False, full_trace=False, stall_gate=False):
        """omega_smd_dir (ADR-017 후보): 깊은 굽힘 펄스의 방향을 SMDD−SMDV 막전위 차의 부호로 정한다 (차 > omega_dir_thresh mV 이면 등쪽, 아니면 배쪽 기본).
        근거: Nature Neurosci 2026 (bioRxiv 2024.08.11.607076) — SMD 가 오메가 진폭, RIV 가 배쪽 편향, SMDD/SMDV 가 등/배 방향."""
        """미로 (H5): walls (S,4) 벽 선분; touch_obs True 면 관측에 (좌, 우, 정면) 접촉 3개 추가; starts [(x, y, heading)], goals [(x, y)] 목록에서 에피소드마다 무작위 선택
        (None 이면 기존 평지 규칙: 원점 시작·무작위 방향, 소스 거리 src_dist·무작위 각)."""
        """보상: 매 스텝 100·ΔC − living_cost (미도달 동안), 도달 +10, 종료 시 −terminal_dist_w·거리. 생존 비용·거리 벌점이 없으면 '정지'(보상 0)가 국소 최적이 된다 (2026-09-03 CPU ES 관측)."""
        self.channels, self.mode, self.dt_action, self.episode_s, self.n_hist, self.amp_max = list(channels), mode, dt_action, episode_s, n_hist, amp_max
        self.src_dist, self.sigma, self.reach_r, self.hidden, self.dtype = src_dist, sigma, reach_r, hidden, dtype
        self.living_cost, self.terminal_dist_w, self.dwell = living_cost, terminal_dist_w, dwell   # dwell: 도달해도 계속, 반경 안 스텝마다 +0.5 (H4 정지 과제)
        self.move_cost = move_cost                                                               # 스텝당 머리 이동 거리(mm) × move_cost 벌점 (에너지 비용; H4-2 변형)
        net = load_network(nml); net.pulses = np.zeros((0, 4)); net, _ = make_variant(net, variant); self.net = net; N = net.n
        self.P = jaxsim.build_params(net, dtype); self.S0 = jaxsim.init_state(self.P)
        self.dt_neural, self.block_ms = dt_neural, block_ms; self.n_neural_steps = int(round(block_ms / dt_neural)); self.n_blocks = int(round(dt_action * 1000 / block_ms))
        self.touch_obs = touch_obs; self.no_odor = no_odor; self.zero_touch = zero_touch; self.proprio_obs = proprio_obs; self.omega_lock = omega_lock; self.full_trace = full_trace; self.stall_gate = stall_gate; self.lateral_obs, self.lateral_d = lateral_obs, lateral_d; self.omega_smd_dir, self.omega_dir_thresh = omega_smd_dir, omega_dir_thresh; self.starts = None if starts is None else np.asarray(starts, float); self.goals = None if goals is None else np.asarray(goals, float)
        self.T = int(round(episode_s / dt_action)); self.n_act = len(self.channels) if mode == "whitelist" else 4; self.n_obs = n_hist + (3 if touch_obs else 0) + (1 if proprio_obs else 0) + (1 if lateral_obs else 0)
        idx = lambda names: jnp.asarray([net.index(n) for n in names])
        self.AVB, self.AVA, self.SMDD, self.SMDV, self.RIV, self.RIS = (idx(CHANNELS[c]) for c in ("AVB", "AVA", "SMDD", "SMDV", "RIV", "RIS"))
        M = np.zeros((self.n_act, N)); self.i_domega = self.channels.index("DOMEGA") if "DOMEGA" in self.channels else None
        if mode == "whitelist":
            for k, c in enumerate(self.channels):
                if c == "DOMEGA": continue                          # H5-6 가상 채널: 등쪽 오메가 드라이브 (뉴런 아님, 원칙 위반을 명시한 변형)
                for n in CHANNELS[c]: M[k, net.index(n)] = amp_max
        self.M_act = jnp.asarray(M, dtype)
        # Worm 기본값 (worm/sim.py)
        self.gate_thresh, self.k_turn, self.head_rows, self.ris_gate = 3.0, 0.03, 12, True
        self.k_omega, self.riv_thresh, self.omega_T, self.omega_sigma = 0.36, 30.0, 2.5, 4.0
        self.rod = Rod2D(Cn=40.0, walls=walls, k_w=k_w, wall_r=wall_r); kw = dict(G_SR=2.0, sr_span_frac=0.25, I_avb_D=0.5, I_avb_V=0.5)
        self.fwd = BoyleMotor(direction=+1, **kw); self.bwd = BoyleMotor(direction=-1, **dict(kw, sr_span_frac=0.5))
        self._mb = motor_block.make_motor_block(self.rod, self.fwd, self.bwd, 5e-3, block_ms, 6.0)
        self.step_fn = jaxsim.make_step(self.P, dt_neural, N)
        self.V_rest = self._warm_rest()
        self.n_params = self.n_obs * hidden + hidden + hidden * self.n_act + self.n_act
        self._rollout = jax.jit(self._make_rollout())

    def _warm_rest(self):
        """Worm 과 같은 휴지 기준: 무자극 2 s (40 블록) 후의 막전위."""
        N = self.net.n; ts = jnp.zeros(self.n_neural_steps, self.dtype); I0 = jnp.zeros((self.n_neural_steps, N), self.dtype)
        run = jax.jit(lambda S: lax.scan(self.step_fn, S, (ts, I0))[0])
        S = self.S0
        for _ in range(40): S = run(S)
        return S.V

    def policy(self, theta, obs):
        nh, h, na = self.n_obs, self.hidden, self.n_act; i = 0
        W1 = theta[i:i + nh * h].reshape(nh, h); i += nh * h; b1 = theta[i:i + h]; i += h
        W2 = theta[i:i + h * na].reshape(h, na); i += h * na; b2 = theta[i:i + na]
        return jax.nn.sigmoid(jnp.tanh(obs @ W1 + b1) @ W2 + b2)

    def _make_rollout(self):
        N = self.net.n; steps = self.n_neural_steps; ts = jnp.zeros(steps, self.dtype); rows = jnp.arange(24, dtype=self.dtype)
        def neural_block(S, I):                                   # I: (N,) 블록 동안 상수
            S, _ = lax.scan(self.step_fn, S, (ts, jnp.broadcast_to(I, (steps, N)))); return S
        def conc(head, src): return jnp.exp(-((head - src) ** 2).sum() / (2 * self.sigma ** 2))
        def one_block(carry, _, I, direct, dom=0.0):
            S, mst, omega_t0, t, osgn = carry
            S = neural_block(S, I); dV = S.V - self.V_rest
            dB, dA = dV[self.AVB].mean(), dV[self.AVA].mean(); dSMD = dV[self.SMDD].mean() - dV[self.SMDV].mean(); dRIS = dV[self.RIS].mean(); dRIV = dV[self.RIV].mean()
            g_f = jnp.where((dB > self.gate_thresh) & (dB > dA), 1.0, 0.0); g_b = jnp.where((dA > self.gate_thresh) & (dA >= dB), 1.0, 0.0)
            dSMDmax = jnp.maximum(dV[self.SMDD].mean(), dV[self.SMDV].mean())
            quiescent = self.ris_gate & (dRIS > self.gate_thresh) & (dRIS > 1.5 * jnp.maximum(jnp.maximum(dB, dA), dSMDmax))
            g_f = jnp.where(quiescent, 0.0, g_f); g_b = jnp.where(quiescent, 0.0, g_b)
            head_bias = self.k_turn * dSMD; omega_bias = -self.k_omega * jnp.maximum(dRIV - self.riv_thresh, 0.0)
            if direct is not None:                                # 상한 대조군: 운동층 직접 조종 (커넥톰 우회)
                g_f = jnp.where(direct[0] > 0.5, 1.0, 0.0); g_b = jnp.where((direct[1] > 0.5) & (g_f == 0), 1.0, 0.0)
                head_bias = (2 * direct[2] - 1) * 1.0; omega_bias = jnp.where(direct[3] > 0.5, -9.0, 0.0)
            if self.omega_smd_dir:                                # ADR-017 후보: RIV 펄스의 방향 = SMDD−SMDV 부호 (등쪽이면 κ>0)
                omega_bias = jnp.where(dSMD > self.omega_dir_thresh, -omega_bias, omega_bias)
            omega_bias = omega_bias + 9.0 * dom                   # H5-6: 등쪽(κ>0) 깊은 굽힘 펄스, 진폭은 RIV 오메가와 같은 크기
            omega_on = omega_bias != 0
            new_fire = omega_on & (omega_t0 < 0)
            osgn = jnp.where(new_fire, jnp.sign(omega_bias), jnp.where(omega_on, osgn, 0.0))          # 발사 순간의 부호를 기억
            if self.omega_lock: omega_bias = jnp.where(omega_on, osgn * jnp.abs(omega_bias), 0.0)    # 펄스 도중 방향 고정 (ADR-017 수정 후보)
            omega_t0 = jnp.where(omega_on & (omega_t0 < 0), t, jnp.where(omega_on, omega_t0, -1.0))
            koff = jnp.zeros(24, self.dtype).at[: self.head_rows].add(head_bias)
            c = -self.omega_sigma + (24 + 2 * self.omega_sigma) * ((t - omega_t0) / 1000) / self.omega_T
            koff = koff + jnp.where(omega_on, omega_bias * jnp.exp(-((rows - c) / self.omega_sigma) ** 2), 0.0)
            koff_j = 0.5 * (koff[1:] + koff[:-1])
            mst, A_D, A_V, kap = self._mb(mst, g_f, g_b, 0.0, koff, koff_j)
            blk = (jnp.sign(omega_bias), mst[0], jnp.stack([dB, dA, dV[self.SMDD].mean(), dV[self.SMDV].mean(), dRIV, dRIS]), jnp.stack([g_f, g_b])) if self.full_trace else jnp.sign(omega_bias)
            return (S, mst, omega_t0, t + self.block_ms, osgn), blk                       # 블록별: 펄스 부호 (−1 배쪽, +1 등쪽, 0 없음) [+ 몸 좌표, ΔV 판독 6개, 게이트 2개]
        from ..body.rod2d import wall_touch
        def make_obs(hist, x, src=None):                           # 실제 감각 뉴런처럼 상대 변화(Weber): 로그 농도 차 × 10, 첫 성분은 로그 수준 (+ 벽 접촉 3)
            lc = jnp.log(jnp.maximum(hist, 1e-4)); o = jnp.concatenate([(lc[-1:] / 2 + 1), 10.0 * (lc[-1] - lc[:-1])[::-1]])
            if self.no_odor: o = jnp.zeros_like(o)                    # H5-3 접촉만 조건
            if self.touch_obs:
                tch = wall_touch(self.rod, x)
                if self.stall_gate:                                                                                # E1 후속: 정면 접촉(정체) 중에는 농도 "하강" 을 보고하지 않는다 (정체≠하강 구별의 감각 게이트)
                    o = jnp.where(tch[2] > 0.4, o.at[1:].set(jnp.maximum(o[1:], 0.0)), o)
                o = jnp.concatenate([o, jnp.zeros_like(tch) if self.zero_touch else tch])   # zero_touch: 접촉 관측 절제 (관측 크기는 유지)
            if self.proprio_obs:                                                                                   # 고유수용: 머리 6 관절 평균 곡률 / 3 (배쪽 = 음수 = 왼쪽)
                o = jnp.concatenate([o, self.rod.curvature(x)[:6].mean()[None] / 3.0])
            if self.lateral_obs:                                                                                   # H5-8: 머리 좌/우 lateral_d 지점의 로그 농도 차 × 10 (머리 흔들기 표본의 대용; 실제 벌레 감각의 이상화)
                hv = x[0] - x[-1]; hv = hv / (jnp.linalg.norm(hv) + 1e-12); nl = jnp.stack([-hv[1], hv[0]])
                cl = jnp.exp(-((x[0] + self.lateral_d * nl - src) ** 2).sum() / (2 * self.sigma ** 2)); cr = jnp.exp(-((x[0] - self.lateral_d * nl - src) ** 2).sum() / (2 * self.sigma ** 2))
                o = jnp.concatenate([o, 10.0 * (jnp.log(jnp.maximum(cl, 1e-4)) - jnp.log(jnp.maximum(cr, 1e-4)))[None]])
            return o
        self.make_obs = make_obs
        def env_step(carry, _, theta, mask):
            S, mst, omega_t0, t, hist, src, R, reached, done, osgn = carry
            obs = make_obs(hist, mst[0], src); a = self.policy(theta, obs) * mask   # mask: 채널 절제 (H4), 기본 1
            I = (a @ self.M_act) if self.mode == "whitelist" else jnp.zeros(N, self.dtype); direct = None if self.mode == "whitelist" else a
            dom = jnp.where(a[self.i_domega] > 0.5, 1.0, 0.0) if self.i_domega is not None else 0.0
            c0 = conc(mst[0][0], src); head0 = mst[0][0]
            (S, mst, omega_t0, t, osgn), blk = lax.scan(lambda cr, x: one_block(cr, x, I, direct, dom), (S, mst, omega_t0, t, osgn), None, length=self.n_blocks)
            osign_blk = blk[0] if self.full_trace else blk
            osign = jnp.where(jnp.any(osign_blk > 0), 1.0, jnp.where(jnp.any(osign_blk < 0), -1.0, 0.0))   # 행동 스텝의 펄스 부호 (등쪽 우선)
            head = mst[0][0]; c1 = conc(head, src); d = jnp.linalg.norm(head - src)
            r = jnp.where(done, 0.0, 100.0 * (c1 - c0) - self.living_cost - self.move_cost * jnp.linalg.norm(head - head0)); hit = (d < self.reach_r) & (~done)
            if self.dwell:
                inside = d < self.reach_r; r = r + jnp.where(hit & (~reached), 10.0, 0.0) + jnp.where(inside, 0.5, 0.0); reached = reached | hit; done_new = done
            else:
                r = r + jnp.where(hit, 10.0, 0.0); reached = reached | hit; done_new = done | hit
            hist = jnp.concatenate([hist[1:], c1[None]])
            hvec = mst[0][0] - mst[0][-1]; hvec = hvec / (jnp.linalg.norm(hvec) + 1e-12)
            body = (blk[1], blk[2], blk[3]) if self.full_trace else head                 # full_trace: 블록별 (몸 (n_blocks,25,2), ΔV (n_blocks,6), 게이트 (n_blocks,2))
            return (S, mst, omega_t0, t, hist, src, R + r, reached, done_new, osgn), (c1, d, a, head, obs, hvec, osign, body)
        def episode(theta, heading, ang, mask, start_xy, goal_xy):
            s = jnp.arange(self.rod.n + 1) * self.rod.ell; x0 = (start_xy[None] + jnp.stack([s * jnp.cos(heading), s * jnp.sin(heading)], 1)).astype(self.dtype)
            src = jnp.where(jnp.isnan(goal_xy[0]), self.src_dist * jnp.stack([jnp.cos(ang), jnp.sin(ang)]), goal_xy).astype(self.dtype)
            mst = motor_block.init_state(x0, self.fwd, self.bwd, self.dtype); c = conc(x0[0], src); hist = jnp.full((self.n_hist,), c, self.dtype)
            carry = (self.S0, mst, jnp.asarray(-1.0, self.dtype), jnp.asarray(0.0, self.dtype), hist, src, jnp.asarray(0.0, self.dtype), jnp.asarray(False), jnp.asarray(False), jnp.asarray(0.0, self.dtype))
            carry, trace = lax.scan(lambda cr, x: env_step(cr, x, theta, mask), carry, None, length=self.T)
            S, mst, omega_t0, t, hist, src, R, reached, done, osgn = carry
            R = R - self.terminal_dist_w * jnp.where(reached, 0.0, jnp.linalg.norm(mst[0][0] - src))
            return {"R": R, "reached": reached, "dist": jnp.linalg.norm(mst[0][0] - src), "src": src, "c": trace[0], "d": trace[1], "a": trace[2], "head": trace[3],
                    "dwell_frac": (trace[1] < self.reach_r).mean(), "obs": trace[4], "heading": trace[5], "omega_sign": trace[6], "body": trace[7]}   # full_trace 면 body = (blk_body, blk_dv, blk_gates)
        return jax.vmap(episode)

    def rollout(self, thetas, seeds, mask=None, start_jitter=0.0):
        """thetas (B, n_params), seeds (B,) int → dict of (B, ...) numpy. 초기 조건은 WormChemEnv.reset(seed) 와 같은 난수 순서 (heading, 소스 각).
        mask: (n_act,) 채널 절제 마스크 (0 = 그 채널 자극 금지), 기본 전부 1."""
        heads, angs, sxy, gxy = [], [], [], []
        for sd in seeds:
            rng = np.random.default_rng(int(sd)); h = rng.uniform(0, 2 * np.pi); a_ = rng.uniform(0, 2 * np.pi); sx, sy = 0.0, 0.0; g = (np.nan, np.nan)
            if self.starts is not None:
                st = self.starts[rng.integers(len(self.starts))]; sx, sy = st[0], st[1]; h = st[2] if len(st) > 2 and not np.isnan(st[2]) else h
                if start_jitter > 0: sx += rng.normal(0, start_jitter * 0.1); sy += rng.normal(0, start_jitter * 0.1); h += rng.normal(0, start_jitter * np.deg2rad(10))   # 시작 흔들기: 위치 ±0.1 mm·jitter, 방향 ±10°·jitter
            if self.goals is not None: g = tuple(self.goals[rng.integers(len(self.goals))])
            heads.append(h); angs.append(a_); sxy.append((sx, sy)); gxy.append(g)
        m = jnp.ones(self.n_act, self.dtype) if mask is None else jnp.asarray(mask, self.dtype)
        masks = jnp.broadcast_to(m, (len(seeds), self.n_act))
        out = self._rollout(jnp.asarray(thetas, self.dtype), jnp.asarray(heads, self.dtype), jnp.asarray(angs, self.dtype), masks,
                            jnp.asarray(sxy, self.dtype), jnp.asarray(gxy, self.dtype))
        return {k: (tuple(np.asarray(x) for x in v) if isinstance(v, (tuple, list)) else np.asarray(v)) for k, v in out.items()}

def pad_theta(th, old_ch, new_ch, nh_old, h, nh_new):
    """작은 채널·관측 집합에서 학습한 정책 θ 를 큰 집합으로 확장: 없던 채널의 출력 가중치 0·편향 −3, 없던 관측의 입력 가중치 0."""
    th = np.asarray(th); W1 = th[:nh_old * h].reshape(nh_old, h); b1 = th[nh_old * h:nh_old * h + h]
    W2 = th[nh_old * h + h:nh_old * h + h + h * len(old_ch)].reshape(h, len(old_ch)); b2 = th[-len(old_ch):]
    W1n = np.zeros((nh_new, h)); W1n[:nh_old] = W1; W2n = np.zeros((h, len(new_ch))); b2n = np.full(len(new_ch), -3.0)
    for j, c in enumerate(new_ch):
        if c in old_ch: i = old_ch.index(c); W2n[:, j] = W2[:, i]; b2n[j] = b2[i]
    return np.concatenate([W1n.ravel(), b1, W2n.ravel(), b2n])

def fit_theta(th, env, old_ch=None):
    """θ 크기가 env 와 다르면 pad_theta 로 맞춘다 (old_ch 기본: env 채널과 같다고 가정)."""
    th = np.asarray(th)
    if th.shape[0] == env.n_params: return th
    old_ch = env.channels if old_ch is None else list(old_ch); h = env.hidden
    nh_old = (th.shape[0] - h - h * len(old_ch) - len(old_ch)) // h
    return pad_theta(th, old_ch, env.channels, nh_old, h, env.n_obs)
