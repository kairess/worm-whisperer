"""2D 점탄성 막대 신체 모델 (docs/BODY_MODEL.md).

- 노드 x_i (i = 0..n), 분절 n개, 체장 L(mm). 벌레는 옆으로 누워 있어 등/배 굽힘이 평면 내 회전.
- 에너지: 신장 (k_s/2)(|Δx| − ℓ)² + 굽힘 (k_b/2)(θ_j − θ0_j)²,  θ0_j = ℓ·κ0_j (근육이 정하는 선호 곡률).
- 과감쇠 동역학 (저 레이놀즈): D_i v_i = −∂E/∂x_i,  D_i = C_t t tᵀ + C_n n nᵀ (저항력 이론, Gray & Lissmann 1964).
  한천 기어가기 C_n/C_t ≈ 40, 수영 ≈ 1.5 (Boyle et al. 2012).
- 단위: 길이 mm, 시간 s. 힘 단위는 C_t = 1 로 정규화.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
from jax import lax

class Rod2D:
    def __init__(self, n_seg=24, L=1.0, Ct=1.0, Cn=40.0, k_s=5e4, k_b=5.0, dt=1e-5, dtype=jnp.float32, walls=None, wall_r=0.06, k_w=5e3):
        """walls: (S, 4) 선분 [x1, y1, x2, y2] (mm) 또는 None. 노드가 벽에서 wall_r 안으로 들어오면 척력 (에너지 ½ k_w (wall_r − d)²). Worm Gym 미로용 (H5)."""
        self.n, self.L, self.ell = n_seg, L, L / n_seg
        self.Ct, self.Cn, self.k_s, self.k_b, self.dt, self.dtype = Ct, Cn, k_s, k_b, dt, dtype
        self.walls = None if walls is None else jnp.asarray(np.asarray(walls, float).reshape(-1, 4), dtype); self.wall_r, self.k_w = wall_r, k_w
        self._run = jax.jit(self._make_run())

    def wall_dist(self, x):
        """각 노드에서 가장 가까운 벽까지의 거리 (n+1,) 와 벽→노드 단위 벡터 (n+1, 2). 벽이 없으면 큰 값."""
        if self.walls is None: return jnp.full((x.shape[0],), 1e3, self.dtype), jnp.zeros_like(x)
        a = self.walls[:, :2]; b = self.walls[:, 2:]; ab = b - a; L2 = jnp.sum(ab ** 2, 1) + 1e-12            # (S,2)
        t = jnp.clip(jnp.sum((x[:, None, :] - a[None]) * ab[None], 2) / L2[None], 0.0, 1.0)                  # (n+1, S)
        proj = a[None] + t[..., None] * ab[None]; diff = x[:, None, :] - proj; d = jnp.linalg.norm(diff, axis=2) + 1e-9
        i = jnp.argmin(d, 1); dmin = jnp.take_along_axis(d, i[:, None], 1)[:, 0]; v = jnp.take_along_axis(diff, i[:, None, None], 1)[:, 0] / dmin[:, None]
        return dmin, v

    def initial(self, x0=(0.0, 0.0), heading=0.0):
        s = np.arange(self.n + 1) * self.ell
        return jnp.asarray(np.stack([x0[0] + s * np.cos(heading), x0[1] + s * np.sin(heading)], 1), self.dtype)

    def energy(self, x, kappa0):
        d = x[1:] - x[:-1]; ln = jnp.linalg.norm(d, axis=1)
        E_s = 0.5 * self.k_s * jnp.sum((ln - self.ell) ** 2)
        phi = jnp.arctan2(d[:, 1], d[:, 0])
        dphi = phi[1:] - phi[:-1]; dphi = jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))   # 관절 각 (n−1)
        E_b = 0.5 * self.k_b * jnp.sum((dphi - self.ell * kappa0) ** 2)
        if self.walls is None: return E_s + E_b
        d, _ = self.wall_dist(x); E_w = 0.5 * self.k_w * jnp.sum(jnp.maximum(self.wall_r - d, 0.0) ** 2)      # 벽 척력
        return E_s + E_b + E_w

    def _make_run(self):
        grad = jax.grad(self.energy)
        def step(x, kappa0):
            F = -grad(x, kappa0)
            d = x[1:] - x[:-1]; t_seg = d / (jnp.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
            t = jnp.concatenate([t_seg[:1], 0.5 * (t_seg[1:] + t_seg[:-1]), t_seg[-1:]], 0)
            t = t / (jnp.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
            Ft = jnp.sum(F * t, 1, keepdims=True) * t; Fn = F - Ft
            v = Ft / self.Ct + Fn / self.Cn
            return x + self.dt * v, None
        def run(x, kappa0_seq):            # kappa0_seq: (steps, n−1)
            x, _ = lax.scan(step, x, kappa0_seq, unroll=8)
            return x
        self._step = lambda x, kappa0: step(x, kappa0)[0]   # motor_block.py 가 한 스텝씩 쓴다
        return run

    def run(self, x, kappa0_seq):
        return self._run(x, jnp.asarray(kappa0_seq, self.dtype))

    def curvature(self, x):
        d = x[1:] - x[:-1]; phi = jnp.arctan2(d[:, 1], d[:, 0]); dphi = phi[1:] - phi[:-1]
        return jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi)) / self.ell

def traveling_wave(rod: Rod2D, A, wavelength, freq, t, phase0=0.0):
    """선호 곡률 진행파 κ0(s,t) = A sin(2π(s/λ − f t)), 관절 위치 s_j = (j+1)ℓ. 머리(s=0)→꼬리로 전파하면 전진."""
    s = (np.arange(rod.n - 1) + 1) * rod.ell
    return A * np.sin(2 * np.pi * (s[None, :] / wavelength - freq * np.asarray(t)[:, None]) + phase0)

def kinematics(rod: Rod2D, xs, dt_sample):
    """궤적 (T, n+1, 2) → 중심 속도(mm/s), 체축 방향 속도, 파동 주파수(Hz, 중간 관절 곡률 FFT)."""
    c = np.asarray(xs).mean(1); v = np.gradient(c, dt_sample, axis=0)
    head_dir = np.asarray(xs)[:, 0] - np.asarray(xs)[:, -1]; head_dir /= np.linalg.norm(head_dir, axis=1, keepdims=True) + 1e-12
    v_axial = np.sum(v * head_dir, 1)
    kap = np.stack([np.asarray(rod.curvature(x)) for x in xs])[:, rod.n // 2]
    f = np.fft.rfftfreq(len(kap), dt_sample); P = np.abs(np.fft.rfft(kap - kap.mean())) ** 2
    return {"speed": float(np.linalg.norm(v, axis=1).mean()), "v_axial": float(v_axial.mean()), "freq": float(f[P[1:].argmax() + 1]) if len(kap) > 4 else np.nan}

def wall_touch(rod: Rod2D, x, n_head=4, r_sense=None):
    """벽 접촉 관측 (H5): 머리 n_head 노드 기준 (좌, 우, 정면) 접촉 강도 ∈ [0,1]. r_sense 안에서 선형 증가 (기본 2·wall_r).
    좌/우는 머리 방향 벡터 h 와 노드→벽 방향의 외적 부호로, 정면은 h 와의 내적으로 가른다."""
    r_sense = 2 * rod.wall_r if r_sense is None else r_sense
    d, v = rod.wall_dist(x[:n_head]); to_wall = -v                                  # 노드에서 벽으로 향하는 단위 벡터
    h = x[0] - x[-1]; h = h / (jnp.linalg.norm(h) + 1e-12)
    strength = jnp.clip(1.0 - d / r_sense, 0.0, 1.0)
    cross = h[0] * to_wall[:, 1] - h[1] * to_wall[:, 0]; dot = h[0] * to_wall[:, 0] + h[1] * to_wall[:, 1]
    left = jnp.max(strength * (cross > 0.3)); right = jnp.max(strength * (cross < -0.3)); front = jnp.max(strength * (dot > 0.7))
    return jnp.stack([left, right, front])

def corridor_walls(points, width):
    """폴리라인 points [(x,y),...] 를 중심선으로 하는 폭 width 의 통로 벽 (양쪽 평행선) 선분 배열. 꺾임점은 단순히 각 구간을 따로 만든다 (모서리 틈 ≤ width/2 는 wall_r 로 메워짐)."""
    segs = []
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        dx, dy = x2 - x1, y2 - y1; L = np.hypot(dx, dy); nx, ny = -dy / L * width / 2, dx / L * width / 2
        segs.append([x1 + nx, y1 + ny, x2 + nx, y2 + ny]); segs.append([x1 - nx, y1 - ny, x2 - nx, y2 - ny])
    return np.array(segs)
