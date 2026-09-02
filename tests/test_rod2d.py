import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worm.body.rod2d import Rod2D, traveling_wave, kinematics

def simulate(Cn=40.0, A=6.0, wavelength=0.9, freq=0.4, T=8.0, block=0.05):
    rod = Rod2D(Cn=Cn); x = rod.initial(); steps = int(block / rod.dt); xs = [x]; t = 0.0
    for b in range(int(T / block)):
        tt = t + np.arange(steps) * rod.dt
        x = rod.run(x, traveling_wave(rod, A, wavelength, freq, tt)); xs.append(x); t += block
    xs = np.stack([np.asarray(v) for v in xs]); return rod, xs[int(2.0 / block):]   # 처음 2 s 과도 제외

def test_crawl_speed_in_literature_range():
    rod, xs = simulate(); k = kinematics(rod, xs, 0.05)
    print(k)
    assert 0.1 <= abs(k["v_axial"]) <= 0.25, k
    assert abs(k["freq"] - 0.4) < 0.15, k   # FFT 해상도 0.17 Hz (6 s 창)
    # 길이 보존
    L = np.linalg.norm(np.diff(xs[-1], axis=0), axis=1).sum(); assert abs(L - 1.0) < 0.03

def test_direction_follows_wave():
    rod, xs = simulate(); k = kinematics(rod, xs, 0.05)
    assert k["v_axial"] > 0          # 머리→꼬리 진행파는 머리 방향으로 전진

def test_swimming_drag_ratio_reduces_thrust_per_cycle():
    _, xs1 = simulate(Cn=40.0); _, xs2 = simulate(Cn=1.5)
    rod = Rod2D(); k1 = kinematics(rod, xs1, 0.05); k2 = kinematics(rod, xs2, 0.05)
    assert abs(k2["v_axial"]) < abs(k1["v_axial"])

def test_symmetric_activation_goes_straight():
    rod, xs = simulate(); c = xs.mean(1); disp = c[-1] - c[0]
    hd = xs[0, 0] - xs[0, -1]                                   # 체축 (꼬리→머리)
    ang = np.arctan2(disp[1], disp[0]) - np.arctan2(hd[1], hd[0]); ang = np.arctan2(np.sin(ang), np.cos(ang))
    assert abs(ang) < 0.26                                      # 진행 방향이 체축에서 15° 이내
