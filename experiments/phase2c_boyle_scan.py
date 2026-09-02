"""Boyle 2012 운동층 + Rod2D: 신장 수용기 이득과 곡률 배율 스캔, 기어가기 운동학 측정."""
import os, sys, itertools, time, numpy as np; sys.path.insert(0, os.getcwd())
from worm.body.rod2d import Rod2D, kinematics
from worm.body.boyle_motor import BoyleMotor
from worm.body.muscle_map import preferred_curvature
def run(G_SR, kappa_max, T=12.0, Cn=40.0, motor_dt=1e-3, I_D=0.5, I_V=0.5, eps=0.5, span=0.5, trace=False):
    rod = Rod2D(Cn=Cn); x = rod.initial(); m = BoyleMotor(G_SR=G_SR, I_avb_D=I_D, I_avb_V=I_V, eps=eps, sr_span_frac=span, init_ventral=True); Slog = []
    steps = int(motor_dt / rod.dt); xs = []; kap = np.zeros(rod.n - 1); t = 0.0
    while t < T:
        kr = np.concatenate([[kap[0]], 0.5 * (kap[1:] + kap[:-1]), [kap[-1]]])      # 관절 곡률 → 행 곡률 (24)
        A_D, A_V = m.step(kr, motor_dt); k0 = preferred_curvature(A_D, A_V, kappa_max)
        x = rod.run(x, np.broadcast_to(k0, (steps, len(k0)))); kap = np.asarray(rod.curvature(x)); t += motor_dt
        if int(round(t / motor_dt)) % 50 == 0: xs.append(np.asarray(x)); Slog.append(m.S[:, 0] - m.S[:, 1])
    if trace:
        for i in range(0, len(Slog), 8): print(f"   t {i*0.05:5.2f}s  D−V state: {''.join('D' if v > 0 else ('V' if v < 0 else '.') for v in Slog[i])}")
    xs = np.stack(xs); k = kinematics(rod, xs[int(4.0 / 0.05):], 0.05)
    K = np.stack([np.asarray(rod.curvature(v)) for v in xs[int(4.0 / 0.05):]])
    # 파장: 시간 평균 공간 자기상관의 첫 최소 위치 ×2
    a, b = K[:, 4] - K[:, 4].mean(), K[:, 14] - K[:, 14].mean()
    lag = (np.argmax(np.correlate(b, a, "full")) - (len(a) - 1)) * 0.05 if a.std() > 1e-6 else np.nan
    return k, K.std(0).mean(), lag, xs
if __name__ == "__main__":
    print("G_SR  kmax  span  speed   v_axial  freq  kappa_std  lag(j4→j14)")
    for G, km, sp in itertools.product([1.0, 2.0, 4.0, 8.0], [6.0, 10.0], [0.5, 0.25]):
        t0 = time.time(); k, ks, lag, _ = run(G, km, span=sp, trace=(G == 2.0 and km == 6.0))
        print(f"{G:4.0f}  {km:4.0f}  {sp:4.2f}  {k['speed']:.4f}  {k['v_axial']:+.4f}  {k['freq']:.2f}  {ks:.2f}  {lag:+.2f}s  ({time.time()-t0:.0f}s)")
