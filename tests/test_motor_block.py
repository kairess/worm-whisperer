"""motor_block (JAX 한 블록) 이 numpy 경로(BoyleMotor + 막대 10회 호출)와 같은 궤적을 내는지."""
import os, sys, numpy as np; sys.path.insert(0, os.getcwd())
from worm.sim import Worm
NML = "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml"
def _run(fast):
    w = Worm(NML, "Vfit", motor="boyle"); w.fast_motor = fast
    sch = [(0, 2, {"AVBL": 5.0, "AVBR": 5.0}), (2, 4, {"AVBL": 5.0, "AVBR": 5.0, "SMDVL": 5.0, "SMDVR": 5.0})]
    w.run_schedule(sch, 4.0); return np.stack([r["x"] for r in w.log]), np.stack([r["kappa"] for r in w.log])
def test_motor_block_matches_numpy_path():
    xs_f, k_f = _run(True); xs_n, k_n = _run(False)
    assert np.abs(xs_f - xs_n).max() < 1e-3, np.abs(xs_f - xs_n).max()      # mm
    assert np.abs(k_f - k_n).max() < 1e-2, np.abs(k_f - k_n).max()          # 1/mm
