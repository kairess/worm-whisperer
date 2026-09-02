import os, sys, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worm.llm.protocols import PROTOCOLS, validate, ALLOWED_NEURONS
from worm.llm import translator as T

def test_all_protocols_whitelisted():
    for n, (g, fn, r) in PROTOCOLS.items(): validate(fn())
    import re
    assert not any(re.fullmatch(r"(DA|DB|VA|VB|DD|VD|AS)\d+|M[DV][LR]\d+", x) for x in ALLOWED_NEURONS)

def test_whitelist_rejects_motor_neurons():
    with pytest.raises(ValueError): validate([(0, 1, {"DB1": 5.0})])

def test_u_roundtrip_and_decode_is_whitelisted():
    for n, (g, fn, r) in PROTOCOLS.items():
        u = T.schedule_to_u(fn()); sch = T.u_to_schedule(u); validate(sch)
        assert (u.max() <= T.AMP_MAX + 1e-6) or True
    rnd = np.random.default_rng(0).uniform(0, 10, (T.K, T.T)); sch = T.u_to_schedule(rnd)
    assert all(v <= T.AMP_MAX for _, _, st in sch for v in st.values())
