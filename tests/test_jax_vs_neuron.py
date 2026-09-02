"""JAX 재구현이 NEURON(c302 C2) 기준 실행과 일치하는지 회귀 테스트.
기준 데이터는 experiments/phase0_c302_reference.py 로 만든 runs/phase0/<ref>/ 를 사용한다 (없으면 skip).
"""
import os, sys
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worm.neural.connectome import load_network, lems_settings
from worm.neural import jaxsim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = ["c302_C2_LW_Full_avb-ava", "c302_C2_LW_Motor_avb-ava", "c302_C2_LW_Cells_AVAL_step"]

def _load_neuron(run_dir, ref, cols):
    d = np.loadtxt(os.path.join(run_dir, f"{ref}.dat")); names = list(cols[f"{ref}.dat"])
    V = d[:, 1:] * 1000
    mf = os.path.join(run_dir, f"{ref}.muscles.dat")
    if os.path.exists(mf):
        V = np.hstack([V, np.loadtxt(mf)[:, 1:] * 1000]); names += cols[f"{ref}.muscles.dat"]
    return V, names

@pytest.mark.parametrize("ref", CASES)
def test_matches_neuron(ref):
    run_dir = os.path.join(ROOT, "runs", "phase0", ref)
    if not os.path.exists(os.path.join(run_dir, f"{ref}.dat")):
        pytest.skip("reference run missing")
    net = load_network(os.path.join(run_dir, ref + ".net.nml"))
    dt, duration, cols = lems_settings(os.path.join(run_dir, f"LEMS_{ref}.xml"))
    P = jaxsim.build_params(net)
    _, Vs = jaxsim.simulate(P, dt, duration, net.n)
    VN, names = _load_neuron(run_dir, ref, cols)
    order = [net.index(n) for n in names]
    VJ = np.vstack([np.asarray(P.v_init)[order][None, :], np.asarray(Vs)[:, order]])
    err = VJ - VN
    rms = np.sqrt((err ** 2).mean())
    assert rms < 0.1, f"RMS {rms} mV"
    assert np.abs(err).max() < 1.0, f"max {np.abs(err).max()} mV"

def test_wormsim_blocks_match_single_run():
    """블록 단위(WormSim.run) 진행이 한 번에 돌린 결과와 같아야 한다 (대화형 인터페이스 검증)."""
    ref = CASES[0]; run_dir = os.path.join(ROOT, "runs", "phase0", ref)
    if not os.path.exists(os.path.join(run_dir, ref + ".net.nml")):
        pytest.skip("reference run missing")
    net = load_network(os.path.join(run_dir, ref + ".net.nml"))
    net.pulses = np.zeros((0, 4))                      # 펄스 제거, 외부 전류는 블록 API로
    P = jaxsim.build_params(net)
    steps = int(200 / 0.05); I = np.zeros((steps, net.n)); I[:, net.index("AVBL")] = 10.0
    _, V1 = jaxsim.simulate(P, 0.05, 200, net.n, I_ext=I)
    sim = jaxsim.WormSim(net)
    V2 = np.vstack([sim.run(50, {"AVBL": 10.0}) for _ in range(4)])
    assert np.abs(np.asarray(V1) - V2).max() < 1e-9
