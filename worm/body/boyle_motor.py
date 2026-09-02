"""Boyle, Berri & Cohen 2012 운동층 (쌍안정 B 뉴런 + 후방 신장 수용기 + 근육 저역통과).

- N 단위(기본 12), 각 단위는 체벽근 행 24/N 개를 담당.
- B 뉴런 상태 S ∈ {0,1}: 입력 I > 0.5(1+ε) 이면 켜지고, I < 0.5(1−ε) 이면 꺼짐 (ε = 0.5 → 0.75 / 0.25).
- 입력 I = I_AVB(측별 상수; 배 1.175, 등 0.675) + I_SR.
- 신장 수용기: 단위 n 의 담당 행부터 후방으로 체장 절반까지의 해당 측 변형률(신장 +, 압축 −) 평균 × G_n,
  G_n 은 머리→꼬리로 선형 증가 (Boyle: (0.224 + 0.056 n)). 우리 단위계에서는 G_SR 배율 하나로 스케일.
- 측 변형률: 등쪽 ε_D = −κ·R, 배쪽 ε_V = +κ·R (R = 체 반지름 0.04 mm; κ > 0 이 등쪽 굽힘).
- 근육: dA/dt = (I_m − A)/τ_M, τ_M = 100 ms, I_m,D = S_D − S_V (D 뉴런 상호 억제, 선형), 클리핑 [0,1].
"""
import numpy as np

class BoyleMotor:
    def __init__(self, n_units=12, n_rows=24, eps=0.5, I_avb_D=0.675, I_avb_V=1.175, G_SR=1.0, R=0.04, tau_M=0.1, sr_span_frac=0.5, inhib=1.0, init_ventral=False, direction=+1):
        """direction +1: 신장 수용기가 후방을 감지 (B형, 머리→꼬리 파동, 전진). −1: 전방 감지 (A형 가정, 꼬리→머리 파동, 후진)."""
        self.direction = direction
        self.N, self.rows = n_units, n_rows; self.per = n_rows // n_units
        self.on, self.off = 0.5 * (1 + eps), 0.5 * (1 - eps)
        self.I_avb = np.array([I_avb_D, I_avb_V]); self.G = G_SR * (0.224 + 0.056 * np.arange(1, n_units + 1)) / 0.224
        self.R, self.tau_M, self.inhib = R, tau_M, inhib
        self.span = max(1, int(round(sr_span_frac * n_rows)))
        self.S = np.zeros((n_units, 2)); self.A = np.zeros((n_rows, 2))   # [:,0]=등, [:,1]=배
        if init_ventral: self.S[:, 1] = 1.0                               # 시동: 배쪽 먼저 (Boyle 의 배쪽 비대칭에 해당)

    def step(self, kappa_rows, dt, avb_gate=1.0, steer=0.0, n_head=3, kappa_offset=None):
        """steer: 머리 단위 입력 편향(사용 안 함, 기본 0). kappa_offset: 신장 수용기의 기준 곡률(조향 편향). 수용기는 (κ − κ_offset) 만 감지
        (Boyle: 변형률은 국소 휴지 길이 기준 → 조향은 휴지 길이 기준의 이동으로 모델링)."""
        if kappa_offset is not None: kappa_rows = kappa_rows - kappa_offset
        """kappa_rows: 행별 곡률 (24,), dt 초, avb_gate: 명령 개재뉴런 게이트 (0–1). 반환 (A_D, A_V) 행별 근육 활성."""
        if avb_gate < 0.5: self.S[:] = 0.0            # 명령 입력이 없으면 B 뉴런 꺼짐 (Boyle: I_AVB 가 회로 전체를 켜고 끔)
        elif not self.S.any(): self.S[:, 1] = 1.0     # 켜질 때 배쪽 시동
        eps_D = -kappa_rows * self.R; eps_V = kappa_rows * self.R
        I = np.zeros((self.N, 2))
        for n in range(self.N):
            if self.direction > 0: a = n * self.per; b = min(self.rows, a + self.span)
            else: b = (n + 1) * self.per; a = max(0, b - self.span)
            I[n, 0] = self.I_avb[0] * avb_gate + self.G[n] * eps_D[a:b].mean()
            I[n, 1] = self.I_avb[1] * avb_gate + self.G[n] * eps_V[a:b].mean()
        I[:n_head, 0] += steer; I[:n_head, 1] -= steer
        self.S = np.where(I > self.on, 1.0, np.where(I < self.off, 0.0, self.S))
        Im = np.stack([self.S[:, 0] - self.inhib * self.S[:, 1], self.S[:, 1] - self.inhib * self.S[:, 0]], 1).clip(0, 1)
        Im_rows = np.repeat(Im, self.per, axis=0)[: self.rows]
        self.A += dt / self.tau_M * (Im_rows - self.A)
        return self.A[:, 0], self.A[:, 1]
