"""커넥톰(Vfit) → Boyle 운동층 결합: 명령 자극으로 방향이 정해진 이동이 나와야 한다."""
import os, sys, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NML = os.path.join(ROOT, "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
pytestmark = pytest.mark.skipif(not os.path.exists(NML), reason="reference network missing")
from worm.sim import Worm, kinematics_from_log

def _run(stim):
    w = Worm(NML, "Vfit", motor="boyle"); w.run(10.0, stim); return kinematics_from_log(w.log, skip_s=4.0)

def test_command_neurons_drive_locomotion():
    k0 = _run({}); kf = _run({"AVBL": 5.0, "AVBR": 5.0}); kb = _run({"AVAL": 5.0, "AVAR": 5.0})
    assert k0["speed"] < 0.01
    assert kf["v_axial"] > 0.1 and 0.25 <= kf["freq"] <= 0.75
    assert kb["v_axial"] < -0.1
