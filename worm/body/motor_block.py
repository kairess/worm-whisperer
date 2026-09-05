"""Boyle 운동층 + 2D 막대를 한 블록(기본 50 ms = 운동층 5 ms × 10)으로 묶은 JAX jit 구현.

worm/sim.py 의 numpy 경로(BoyleMotor.step 을 5 ms 마다 호출, 막대 run 을 10번 호출)와 같은 계산을 하나의 컴파일된 함수로 만든다.
목적: (1) JAX↔numpy 변환 제거로 대화형 시뮬레이션 3–4배 가속, (2) vmap 으로 환경 수백 개를 GPU 에서 동시에 돌리는 Worm Gym 배치 시뮬레이터의 토대.
상태: x (n+1, 2) 막대 좌표, S_f/A_f (전진 회로 뉴런 상태·근육 활성), S_b/A_b (후진 회로). 수치는 numpy 경로와 float32 오차 안에서 같다 (tests/test_motor_block.py).
"""
from __future__ import annotations
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from .rod2d import Rod2D
from .boyle_motor import BoyleMotor

def _span_matrix(m: BoyleMotor):
    """단위 n 의 신장 수용기 평균 창을 (N, rows) 행렬로: mean(eps[a:b]) = M @ eps."""
    M = np.zeros((m.N, m.rows))
    for n in range(m.N):
        if m.direction > 0: a = n * m.per; b = min(m.rows, a + m.span)
        else: b = (n + 1) * m.per; a = max(0, b - m.span)
        M[n, a:b] = 1.0 / (b - a)
    return M

def make_motor_block(rod: Rod2D, fwd: BoyleMotor, bwd: BoyleMotor, motor_dt=5e-3, block_ms=50.0, kappa_max=6.0, head_units=3):
    n_sub = int(round(block_ms / 1000 / motor_dt)); rod_steps = int(round(motor_dt / rod.dt)); dt = jnp.asarray(motor_dt, rod.dtype)
    consts = []
    for m in (fwd, bwd):
        consts.append(dict(M=jnp.asarray(_span_matrix(m), rod.dtype), G=jnp.asarray(m.G, rod.dtype), I_avb=jnp.asarray(m.I_avb, rod.dtype),
                           on=m.on, off=m.off, R=m.R, tau=m.tau_M, inhib=m.inhib, per=m.per, rows=m.rows))
    def boyle_step(c, S, A, kappa_rows, gate, steer):
        S = jnp.where(gate < 0.5, 0.0, S)
        S = jnp.where((gate >= 0.5) & (S.sum() == 0), S.at[:, 1].set(1.0), S)          # 켜질 때 배쪽 시동
        eps_D = -kappa_rows * c["R"]; eps_V = kappa_rows * c["R"]
        I = jnp.stack([c["I_avb"][0] * gate + c["G"] * (c["M"] @ eps_D), c["I_avb"][1] * gate + c["G"] * (c["M"] @ eps_V)], 1)
        I = I.at[:head_units, 0].add(steer).at[:head_units, 1].add(-steer)
        S = jnp.where(I > c["on"], 1.0, jnp.where(I < c["off"], 0.0, S))
        Im = jnp.stack([S[:, 0] - c["inhib"] * S[:, 1], S[:, 1] - c["inhib"] * S[:, 0]], 1).clip(0, 1)
        Im_rows = jnp.repeat(Im, c["per"], axis=0)[: c["rows"]]
        A = A + dt / c["tau"] * (Im_rows - A)
        return S, A
    def rod_run(x, kappa0):
        return lax.fori_loop(0, rod_steps, lambda i, xx: rod._step(xx, kappa0), x)
    def substep(state, _, g_f, g_b, steer, koff, koff_j):
        x, S_f, A_f, S_b, A_b = state
        k = rod.curvature(x); kr = jnp.concatenate([k[:1], 0.5 * (k[1:] + k[:-1]), k[-1:]]) - koff
        S_f, A_f = boyle_step(consts[0], S_f, A_f, kr, g_f, steer); S_b, A_b = boyle_step(consts[1], S_b, A_b, kr, g_b, steer)
        A_D = jnp.clip(A_f[:, 0] + A_b[:, 0], 0, 1); A_V = jnp.clip(A_f[:, 1] + A_b[:, 1], 0, 1)
        diff = kappa_max * (A_D - A_V); kappa0 = 0.5 * (diff[1:] + diff[:-1]) + koff_j
        x = rod_run(x, kappa0.astype(rod.dtype))
        return (x, S_f, A_f, S_b, A_b), (A_D, A_V)
    @jax.jit
    def block(state, g_f, g_b, steer, koff, koff_j):
        state, (A_D, A_V) = lax.scan(lambda s, _: substep(s, _, g_f, g_b, steer, koff, koff_j), state, None, length=n_sub)
        return state, A_D[-1], A_V[-1], rod.curvature(state[0])
    return block

def init_state(x, fwd: BoyleMotor, bwd: BoyleMotor, dtype):
    return (jnp.asarray(x, dtype), jnp.asarray(fwd.S, dtype), jnp.asarray(fwd.A, dtype), jnp.asarray(bwd.S, dtype), jnp.asarray(bwd.A, dtype))
