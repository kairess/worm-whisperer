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
    def __init__(self, n_seg=24, L=1.0, Ct=1.0, Cn=40.0, k_s=5e4, k_b=5.0, dt=1e-5, dtype=jnp.float32):
        self.n, self.L, self.ell = n_seg, L, L / n_seg
        self.Ct, self.Cn, self.k_s, self.k_b, self.dt, self.dtype = Ct, Cn, k_s, k_b, dt, dtype
        self._run = jax.jit(self._make_run())

    def initial(self, x0=(0.0, 0.0), heading=0.0):
        s = np.arange(self.n + 1) * self.ell
        return jnp.asarray(np.stack([x0[0] + s * np.cos(heading), x0[1] + s * np.sin(heading)], 1), self.dtype)

    def energy(self, x, kappa0):
        d = x[1:] - x[:-1]; ln = jnp.linalg.norm(d, axis=1)
        E_s = 0.5 * self.k_s * jnp.sum((ln - self.ell) ** 2)
        phi = jnp.arctan2(d[:, 1], d[:, 0])
        dphi = phi[1:] - phi[:-1]; dphi = jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))   # 관절 각 (n−1)
        E_b = 0.5 * self.k_b * jnp.sum((dphi - self.ell * kappa0) ** 2)
        return E_s + E_b

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
