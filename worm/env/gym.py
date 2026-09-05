"""Worm Gym: 고정 커넥톰 벌레를 화이트리스트 자극만으로 조종하는 강화학습 환경 (docs/PLAN_WORMGYM.md 1절).

관측: 코 끝 농도의 로그 수준 1개 + 로그 농도의 최근 변화 4개(0.5–2.0 s 전 대비, ×10; Weber 법칙형 상대 감각). 나침반·좌표 없음.
  (처음엔 농도 원값 5개를 줬는데 변화량이 0.01 수준이라 정책이 감각을 무시하고 상수 회전 정책으로 퇴화했다 — 2026-09-03.)
행동: 허용 채널마다 자극 진폭 a ∈ [0, 1] → a × amp_max pA, 0.5 s 유지.
보상: 농도 증가분 × 100 − 생존 비용, 도달(reach_r 이내) 시 +10 후 종료, 미도달 종료 시 −2·거리.
mode "whitelist": 채널 = protocols.CHANNELS 의 이름. mode "direct": 운동층 직접 조종 (g_f, g_b, head_bias, omega) — 상한 대조군, 커넥톰 우회.
"""
from __future__ import annotations
import numpy as np
from ..sim import Worm
from ..llm.protocols import CHANNELS
from .chem import ChemField

NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
_V_REST = {}

class WormChemEnv:
    def __init__(self, channels=("AVB", "AVA", "SMDD", "SMDV", "RIV"), mode="whitelist", dt_action=0.5, episode_s=40.0, n_hist=5,
                 amp_max=6.0, src_dist=2.5, sigma=2.0, reach_r=0.5, variant="Vfit", nml=NML, living_cost=0.05, terminal_dist_w=2.0):
        self.channels, self.mode, self.dt_action, self.episode_s, self.n_hist = list(channels), mode, dt_action, episode_s, n_hist
        self.amp_max, self.src_dist, self.sigma, self.reach_r, self.variant, self.nml = amp_max, src_dist, sigma, reach_r, variant, nml
        self.living_cost, self.terminal_dist_w = living_cost, terminal_dist_w    # batch.py 와 같은 보상 성형
        for c in self.channels: assert c in CHANNELS, c
        self.n_act = len(self.channels) if mode == "whitelist" else 4
        self.n_obs = n_hist
        self.w = None

    def _new_worm(self):
        w = Worm(self.nml, self.variant, motor="boyle")
        key = (self.nml, self.variant)
        if key in _V_REST: w.V_rest = _V_REST[key].copy()
        return w

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.w = self._new_worm(); heading = rng.uniform(0, 2 * np.pi)
        self.w.x = self.w.body.initial((0.0, 0.0), heading)          # 머리 x[0] 가 원점, 몸은 heading 방향으로 뻗음 → 진행 방향은 heading + π
        ang = rng.uniform(0, 2 * np.pi); self.field = ChemField((self.src_dist * np.cos(ang), self.src_dist * np.sin(ang)), sigma=self.sigma)
        self.t = 0.0; self.hist = [self._conc()] * self.n_hist; self.done = False; self.reached = False; self.trace = []
        return self._obs()

    def _conc(self): return float(self.field.conc(np.asarray(self.w.x)[0]))
    def _obs(self):                                                  # batch.py 의 make_obs 와 동일: [log C 수준, 10·Δlog C (0.5 … 2.0 s 전 대비)]
        lc = np.log(np.maximum(np.asarray(self.hist[-self.n_hist:], float), 1e-4))
        return np.concatenate([[lc[-1] / 2 + 1], 10.0 * (lc[-1] - lc[:-1])[::-1]]).astype(np.float32)
    def head(self): return np.asarray(self.w.x)[0]
    def dist(self): return float(np.linalg.norm(self.head() - self.field.src))

    def step(self, action):
        a = np.clip(np.asarray(action, float), 0, 1)
        if self.mode == "whitelist":
            I = {}
            for c, ai in zip(self.channels, a):
                for n in CHANNELS[c]: I[n] = I.get(n, 0.0) + float(ai) * self.amp_max
            self.w.override = None
        else:                                                       # direct: [g_f, g_b, head_bias(−1..1), omega]
            g_f = 1.0 if a[0] > 0.5 else 0.0; g_b = 1.0 if (a[1] > 0.5 and not g_f) else 0.0
            self.w.override = {"g_f": g_f, "g_b": g_b, "head_bias": (2 * a[2] - 1) * 1.0, "omega_bias": -9.0 if a[3] > 0.5 else 0.0}; I = {}
        c0 = self._conc(); n_blocks = int(round(self.dt_action * 1000 / self.w.block_ms))
        for _ in range(n_blocks): self.w.step(I)
        key = (self.nml, self.variant)
        if key not in _V_REST and self.w.V_rest is not None: _V_REST[key] = np.asarray(self.w.V_rest).copy()
        self.t += self.dt_action; c1 = self._conc(); self.hist.append(c1)
        r = 100.0 * (c1 - c0) - self.living_cost; d = self.dist()
        if d < self.reach_r: r += 10.0; self.done = True; self.reached = True
        if self.t >= self.episode_s - 1e-9: self.done = True
        if self.done and not self.reached: r -= self.terminal_dist_w * d
        x = np.asarray(self.w.x); hd = x[0] - x[-1]
        self.trace.append({"t": self.t, "c": c1, "d": d, "a": a.copy(), "head": self.head().copy(), "heading": (hd / (np.linalg.norm(hd) + 1e-12)).copy()})
        return self._obs(), r, self.done, {"dist": d, "reached": self.reached}

class MLPPolicy:
    """관측(n_obs) → 은닉 tanh → 시그모이드 행동(n_act). 진화 전략용 평탄 파라미터."""
    def __init__(self, n_obs, n_act, hidden=16):
        self.n_obs, self.n_act, self.h = n_obs, n_act, hidden
        self.n_params = n_obs * hidden + hidden + hidden * n_act + n_act
    def unpack(self, theta):
        i = 0; W1 = theta[i:i + self.n_obs * self.h].reshape(self.n_obs, self.h); i += self.n_obs * self.h
        b1 = theta[i:i + self.h]; i += self.h; W2 = theta[i:i + self.h * self.n_act].reshape(self.h, self.n_act); i += self.h * self.n_act
        b2 = theta[i:i + self.n_act]; return W1, b1, W2, b2
    def act(self, theta, obs):
        W1, b1, W2, b2 = self.unpack(theta); x = np.asarray(obs, float)
        x = np.concatenate([x, np.diff(x)]) if False else x
        h = np.tanh(x @ W1 + b1); return 1 / (1 + np.exp(-(h @ W2 + b2)))

def run_episode(env: WormChemEnv, policy: MLPPolicy | None, theta, seed, fixed_action=None):
    obs = env.reset(seed); R = 0.0; rng = np.random.default_rng(seed + 12345)
    while not env.done:
        if fixed_action == "random": a = rng.uniform(0, 1, env.n_act)
        elif fixed_action is not None: a = np.asarray(fixed_action, float)
        else: a = policy.act(theta, obs)
        obs, r, done, info = env.step(a); R += r
    return {"R": R, "reached": env.reached, "dist": env.dist(), "t": env.t, "c_final": env.hist[-1]}
