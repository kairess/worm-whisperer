"""배치 환경(GPU/JAX 롤아웃)이 numpy 환경(WormChemEnv + Worm)과 같은 에피소드를 내는지: 전진만 정책, seed 0."""
import os, sys, numpy as np; sys.path.insert(0, os.getcwd())
def test_batch_env_matches_numpy_env():
    from worm.env.gym import WormChemEnv, run_episode
    from worm.env.batch import BatchWormEnv
    env = WormChemEnv(episode_s=10.0); r = run_episode(env, None, None, 0, fixed_action=[1, 0, 0, 0, 0])
    benv = BatchWormEnv(episode_s=10.0); theta = np.zeros(benv.n_params); theta[-benv.n_act:] = [30, -30, -30, -30, -30]
    out = benv.rollout(theta[None], [0])
    assert abs(float(out["dist"][0]) - r["dist"]) < 0.05, (out["dist"], r["dist"])
    assert abs(float(out["R"][0]) - r["R"]) < 1.0, (out["R"], r["R"])
