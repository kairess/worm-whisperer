"""E9: 주장 5 의 에피소드 단위 통계. 펄스 단위 일치율(0.73 vs 0.62, n≈5–7천)은 펄스가 에피소드 안에서 독립이 아니므로 과대 신뢰. 여기서는 에피소드마다
실현 펄스 방향 = 소스 쪽 일치율을 구하고 (1) 정책별 평균 ± sd, 부호검정(> 0.5), (2) 좌우 정보 정책 vs ADR-017 정책의 Mann–Whitney U 로 비교한다.
판정(사전 등록, PAPER_OUTLINE §7 E9): 좌우 정보 정책의 에피소드 평균 > 0.5 이고 부호검정 p < 0.01, 두 정책 Mann–Whitney p < 0.01 이면 주장 5 유지."""
import os, sys, json, numpy as np; sys.path.insert(0, os.getcwd()); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from scipy import stats
from worm.env.batch import BatchWormEnv, fit_theta
POL = {"lateral": ("runs/wormgym/h5/lateral/flat/theta_final.npy", dict(lateral_obs=True)), "adr017_nolateral": ("runs/wormgym/h5/smddir/flat_far/theta_final.npy", {})}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 128; res = {}; per = {}
for name, (path, kw) in POL.items():
    env = BatchWormEnv(episode_s=40.0, omega_smd_dir=True, **kw); th = fit_theta(np.load(path), env)
    out = env.rollout(np.stack([th] * n), range(95000, 95000 + n)); H, HV, SRC = out["head"], out["heading"], out["src"]
    lat = HV[:, :-1, 0] * (SRC[:, None, 1] - H[:, :-1, 1]) - HV[:, :-1, 1] * (SRC[:, None, 0] - H[:, :-1, 0])
    OS = np.asarray(out["omega_sign"])[:, 1:]; mo = (OS != 0) & (np.abs(lat) > 0.2); hit = (OS == -np.sign(lat)) & mo
    k = mo.sum(1); ep = np.where(k > 0, hit.sum(1) / np.maximum(k, 1), np.nan); ep = ep[k > 0]; per[name] = ep
    above = int((ep > 0.5).sum()); below = int((ep < 0.5).sum()); p_sign = stats.binomtest(above, above + below, 0.5, alternative="greater").pvalue
    res[name] = {"episodes_with_pulses": int(len(ep)), "pulses_per_episode_median": float(np.median(k[k > 0])), "episode_mean": float(ep.mean()), "episode_sd": float(ep.std(ddof=1)),
                 "episodes_above_0.5": above, "episodes_below_0.5": below, "sign_test_p": float(p_sign), "pooled_pulse_rate": float(hit.sum() / mo.sum()), "reach": float(np.asarray(out["reached"]).mean())}
    print(name, json.dumps(res[name]), flush=True)
u = stats.mannwhitneyu(per["lateral"], per["adr017_nolateral"], alternative="greater"); res["mannwhitney_lateral_gt_none"] = {"U": float(u.statistic), "p": float(u.pvalue)}
d = (per["lateral"].mean() - per["adr017_nolateral"].mean()) / np.sqrt(0.5 * (per["lateral"].var(ddof=1) + per["adr017_nolateral"].var(ddof=1))); res["cohen_d"] = float(d)
ok = res["lateral"]["episode_mean"] > 0.5 and res["lateral"]["sign_test_p"] < 0.01 and u.pvalue < 0.01
res["verdict"] = "KEEP" if ok else "REVISE"; print("verdict", res["verdict"], f"MWU p {u.pvalue:.2e} d {d:.2f}", flush=True)
json.dump(res, open("runs/wormgym/paper/weathervane_episodes.json", "w"), indent=1)
