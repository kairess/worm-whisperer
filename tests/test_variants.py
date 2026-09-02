import os, sys, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worm.neural.connectome import load_network, ablate
from worm.neural.variants import make_variant, apply_theta
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NML = os.path.join(ROOT, "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
pytestmark = pytest.mark.skipif(not os.path.exists(NML), reason="reference network missing")

def test_variants_preserve_connectome():
    net = load_network(NML)
    for name in ["V1", "V1-split", "V2-0.1-gj0.3", "V3-split-0.3-vthm40-leak10"]:
        v, info = make_variant(net, name)
        pairs0 = set(zip(net.syn_pre.tolist(), net.syn_post.tolist())); pairs1 = set(zip(v.syn_pre.tolist(), v.syn_post.tolist()))
        assert pairs0 == pairs1, name                       # 연결의 존재는 불변
        # 쌍별 총 시냅스 개수(가중치 합) 불변
        w0 = {}; w1 = {}
        for p, q, w in zip(net.syn_pre, net.syn_post, net.syn_w): w0[(p, q)] = w0.get((p, q), 0) + w
        for p, q, w in zip(v.syn_pre, v.syn_post, v.syn_w): w1[(p, q)] = w1.get((p, q), 0) + w
        assert all(abs(w0[k] - w1[k]) < 1e-9 for k in w0), name
        assert len(v.gj_a) == len(net.gj_a)

def test_polarity_stats():
    net = load_network(NML); v, info = make_variant(net, "V1")
    assert info["flipped_to_inh"] > 200 and info["flipped_to_exc"] < 50
    assert sum("inh" in s for s in v.syn_id) == sum("inh" in s for s in net.syn_id) + info["flipped_to_inh"] - info["flipped_to_exc"]

def test_theta_zero_is_identity_except_vth_delta():
    net = load_network(NML); v = apply_theta(net, [0, 0, 0, 0, 0.0, 0])
    assert np.allclose(v.syn_g, net.syn_g) and np.allclose(v.gj_g, net.gj_g) and v.leak_scale == 1.0

def test_ablate_removes_all_connections_of_cell():
    net = load_network(NML); i = net.index("AVAL"); v = ablate(net, ["AVAL"])
    assert not np.any(v.syn_pre == i) and not np.any(v.syn_post == i) and not np.any(v.gj_a == i) and not np.any(v.gj_b == i)
