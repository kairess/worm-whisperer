"""H5 미로 무대 (mm, 체장 1 mm 기준). 미세 격자(셀 = 통로 폭)의 열린 셀 집합으로 통로를 정의하고, 열린 셀과 닫힌 셀(또는 바깥) 사이 변을 벽으로 만든다 → 틈이 없다.
각 무대: walls (S,4), starts [(x, y, heading)], goals [(x, y)]. heading 은 몸이 뻗는 방향이고 진행 방향은 heading + π (머리 x[0] 가 시작점).
"""
import numpy as np

def lattice_walls(open_cells, cell):
    """open_cells: {(i, j)} 열린 셀 (중심 (i·cell, j·cell)). 열린 셀의 네 변 중 이웃이 닫힌 변만 벽."""
    segs = []; h = cell / 2
    for (i, j) in open_cells:
        cx, cy = i * cell, j * cell
        if (i + 1, j) not in open_cells: segs.append([cx + h, cy - h, cx + h, cy + h])
        if (i - 1, j) not in open_cells: segs.append([cx - h, cy - h, cx - h, cy + h])
        if (i, j + 1) not in open_cells: segs.append([cx - h, cy + h, cx + h, cy + h])
        if (i, j - 1) not in open_cells: segs.append([cx - h, cy - h, cx + h, cy - h])
    return np.array(segs)

def _line_cells(a, b):
    """격자점 a→b (축 정렬) 사이의 셀 목록 (양 끝 포함)."""
    (i1, j1), (i2, j2) = a, b; cells = []
    if i1 == i2: cells = [(i1, j) for j in range(min(j1, j2), max(j1, j2) + 1)]
    else: cells = [(i, j1) for i in range(min(i1, i2), max(i1, i2) + 1)]
    return cells

def corridor_turn(width=0.3, leg=2.4, mirror=False):
    """L 자 통로: 원점에서 +x 로 leg, 그 뒤 +y 로 leg (왼쪽 회전). mirror=True 면 −y (오른쪽 회전). 시작: 첫 다리 안쪽 (몸이 −x 로 뻗음, 진행 +x). 목표: 두 번째 다리 끝."""
    n = int(round(leg / width)); sgn = -1 if mirror else 1; cells = set(_line_cells((0, 0), (n, 0))) | set(_line_cells((n, 0), (n, sgn * n)))
    return dict(walls=lattice_walls(cells, width), starts=[(1.1, 0.0, np.pi)], goals=[(n * width, sgn * n * width)], name=f"corridor{'_R' if mirror else ''}_w{width}", cells=cells, cell=width)

def corridor_turn_right(width=0.3, leg=2.4): return corridor_turn(width, leg, mirror=True)

def t_maze(width=0.3, stem=3.0, arm=2.1):
    """T-미로: 줄기 (0, −stem)→(0, 0), 팔 (−arm, 0)→(arm, 0). 시작: 줄기 안 (몸이 −y 로 뻗음, 진행 +y). 목표: 두 팔 끝 중 무작위."""
    ns, na = int(round(stem / width)), int(round(arm / width))
    cells = set(_line_cells((0, -ns), (0, 0))) | set(_line_cells((-na, 0), (na, 0)))
    return dict(walls=lattice_walls(cells, width), starts=[(0.0, -stem + 1.1, -np.pi / 2)], goals=[(-na * width, 0.0), (na * width, 0.0)], name=f"tmaze_w{width}", cells=cells, cell=width)

def grid_maze(width=0.3, pitch=1.2, edges=None):
    """3×3 노드 미로: 노드 (a, b) ∈ {0,1,2}², 노드 간격 pitch, 통로 폭 width. edges: 열린 노드 쌍. 시작 노드 (0,0) (몸이 −x 로 뻗도록 통로를 −x 로 1.2 mm 연장), 목표 노드 (2,2)."""
    if edges is None: edges = [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (0, 2)), ((0, 1), (0, 2)), ((1, 2), (2, 2))]
    k = int(round(pitch / width)); cells = set()
    for (a, b) in edges: cells |= set(_line_cells((a[0] * k, a[1] * k), (b[0] * k, b[1] * k)))
    cells |= set(_line_cells((-4, 0), (0, 0)))                                                         # 시작 꼬리 통로
    return dict(walls=lattice_walls(cells, width), starts=[(0.0, 0.0, np.pi)], goals=[(2 * pitch, 2 * pitch)], name=f"grid3_w{width}", cells=cells, cell=width)

MAZES = {"corridor": corridor_turn, "corridor_R": corridor_turn_right, "tmaze": t_maze, "grid": grid_maze}
