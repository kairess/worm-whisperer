"""가상 화학 환경: 2D 가우시안 농도장. 머리 위치의 농도 변화율을 ASE 감각뉴런 전류로 변환 (S 등급 프로토콜).
근거: ASEL 은 NaCl 농도 증가(ON), ASER 은 감소(OFF)에 반응 (Suzuki et al. 2008); 피루엣/클리노택시스는 dC/dt 에 의존 (Pierce-Shimomura 1999; Iino & Yoshida 2009).
"""
import numpy as np

class ChemField:
    def __init__(self, source_xy=(0.0, 2.0), sigma=2.0, peak=1.0):
        self.src = np.asarray(source_xy, float); self.sigma = sigma; self.peak = peak
    def conc(self, xy):
        d2 = ((np.asarray(xy) - self.src) ** 2).sum(-1); return self.peak * np.exp(-d2 / (2 * self.sigma ** 2))

class ASEInput:
    """dC/dt 를 저역통과(τ)한 뒤 ASEL 에는 증가분, ASER 에는 감소분을 전류(pA)로. gain: pA per (농도/s)."""
    def __init__(self, field: ChemField, gain=50.0, tau=0.5):
        self.f, self.gain, self.tau = field, gain, tau; self.prev = None; self.dcdt = 0.0
    def currents(self, head_xy, dt):
        c = float(self.f.conc(head_xy))
        if self.prev is not None:
            self.dcdt += (dt / self.tau) * ((c - self.prev) / dt - self.dcdt)
        self.prev = c
        on, off = max(self.dcdt, 0.0) * self.gain, max(-self.dcdt, 0.0) * self.gain
        return {"ASEL": on, "ASER": off}, c
