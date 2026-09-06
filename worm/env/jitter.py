"""미로 시작 자세 흔들림 (E10). 통로 폭 0.3 mm 에서 위치 σ 0.1 mm·방향 σ 10° 흔들림은 몸이 벽 밖에서 시작하게 만들고, 냄새는 벽을 통과하므로 벽 밖 도달이 집계된다
(2026-09-07 발견). 여기서는 몸 25 마디가 모두 열린 셀 안에 있는 자세만 받아들인다: 통로 축 방향 ±along, 옆 방향 σ lateral, 방향 σ heading_deg."""
import numpy as np

def body_nodes(pose, n_seg=24, L=1.0):
    x, y, h = pose; s = np.arange(n_seg + 1) * (L / n_seg); return np.stack([x + s * np.cos(h), y + s * np.sin(h)], 1)

def inside_cells(pts, cells, cell):
    return all((int(np.floor(px / cell + 0.5)), int(np.floor(py / cell + 0.5))) in cells for px, py in pts)

def valid_poses(mz, n, seed=0, along=0.3, lateral=0.03, heading_deg=3.0, max_tries=20000):
    """mz: MAZES[...](width) 결과 (cells, cell, starts). 반환 (n, 3) 배열; 기본 시작 자세 주변에서 거절 표본."""
    rng = np.random.default_rng(seed); cells, cell = mz["cells"], mz["cell"]; out = []; tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1; x0, y0, h0 = mz["starts"][rng.integers(len(mz["starts"]))]
        u = rng.uniform(-along, along); v = rng.normal(0, lateral); dh = rng.normal(0, np.deg2rad(heading_deg))
        tx, ty = np.cos(h0), np.sin(h0); pose = (x0 + u * tx - v * ty, y0 + u * ty + v * tx, h0 + dh)
        if inside_cells(body_nodes(pose), cells, cell): out.append(pose)
    if len(out) < n: raise RuntimeError(f"only {len(out)} valid poses in {tries} tries")
    return np.array(out)
