"""c302 근육 이름 ↔ (행, 등/배) 매핑과 근육 활성 → 선호 곡률 변환."""
import re, numpy as np

def muscle_rows(names):
    """names: 세포 이름 목록. 반환 dorsal_idx[(24, ≤2)], ventral_idx 리스트(행별 인덱스 목록)."""
    D = [[] for _ in range(24)]; V = [[] for _ in range(24)]
    for i, n in enumerate(names):
        m = re.fullmatch(r"M([DV])([LR])(\d\d)", n)
        if m:
            (D if m.group(1) == "D" else V)[int(m.group(3)) - 1].append(i)
    return D, V

def activation_from_ca(ca, K_half):
    return ca / (ca + K_half)

def preferred_curvature(A_D, A_V, kappa_max):
    """행별 활성(24,) → 관절 선호 곡률(23,): 인접 행 평균. 등쪽 활성 → +κ."""
    diff = kappa_max * (A_D - A_V)
    return 0.5 * (diff[1:] + diff[:-1])
