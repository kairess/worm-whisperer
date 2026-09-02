import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase2c_boyle_scan import run

def test_forward_crawl_in_literature_range():
    k, kstd, lag, xs = run(2.0, 6.0, span=0.25, T=12.0)
    assert 0.1 <= k["v_axial"] <= 0.35, k          # 문헌 0.1–0.25 mm/s (상한 여유)
    assert 0.25 <= k["freq"] <= 0.75, k            # 문헌 0.3–0.5 Hz
    assert lag < 0                                 # 머리→꼬리 전파

def test_no_command_no_motion():
    from worm.body.rod2d import Rod2D, kinematics
    from worm.body.boyle_motor import BoyleMotor
    from worm.body.muscle_map import preferred_curvature
    rod = Rod2D(); x = rod.initial(); m = BoyleMotor(G_SR=2.0, sr_span_frac=0.25, I_avb_D=0.5, I_avb_V=0.5); steps = int(1e-3 / rod.dt); xs = []
    for i in range(4000):
        kr = np.zeros(24); A_D, A_V = m.step(kr, 1e-3, avb_gate=0.0); k0 = preferred_curvature(A_D, A_V, 6.0)
        x = rod.run(x, np.broadcast_to(k0, (steps, 23)))
        if i % 50 == 0: xs.append(np.asarray(x))
    k = kinematics(rod, np.stack(xs), 0.05); assert k["speed"] < 0.01
