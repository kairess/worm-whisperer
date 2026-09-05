"""H2-2 / H5-7 판정 (배치 환경): 평지에서 (a) 조향 부호(SMDV−SMDD, +=왼쪽) 와 소스 쪽 부호의 일치율, (b) RIV 펄스 스텝의 굽힘 방향(SMDD−SMDV 부호, +=등쪽=오른쪽) 과 소스 쪽의 일치율.
사용: uv run python experiments/wormgym_weathervane.py <policy.npy> [--omega_smd_dir] [--proprio] [--episodes 128]"""
import os, sys, json, argparse, numpy as np; sys.path.insert(0, os.getcwd()); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ap = argparse.ArgumentParser(); ap.add_argument("policy"); ap.add_argument("--omega_smd_dir", action="store_true"); ap.add_argument("--proprio", action="store_true"); ap.add_argument("--lateral", action="store_true"); ap.add_argument("--episodes", type=int, default=128)
ap.add_argument("--channels", default="AVB,AVA,SMDD,SMDV,RIV"); ap.add_argument("--variant", default="Vfit"); a = ap.parse_args(); ch = a.channels.split(","); ci = {c: i for i, c in enumerate(ch)}
from worm.env.batch import BatchWormEnv, fit_theta
env = BatchWormEnv(variant=a.variant, channels=ch, episode_s=40.0, omega_smd_dir=a.omega_smd_dir, proprio_obs=a.proprio, lateral_obs=a.lateral)
th = fit_theta(np.load(a.policy), env); out = env.rollout(np.stack([th] * a.episodes), range(95000, 95000 + a.episodes))
H, A, HV, SRC = out["head"], out["a"], out["heading"], out["src"]
lat = HV[:, :-1, 0] * (SRC[:, None, 1] - H[:, :-1, 1]) - HV[:, :-1, 1] * (SRC[:, None, 0] - H[:, :-1, 0])          # + : 소스가 왼쪽 (t 스텝의 자세 → t+1 행동)
Anext = A[:, 1:, :]; steer = Anext[:, :, ci["SMDV"]] - Anext[:, :, ci["SMDD"]]; m = (np.abs(steer) > 0.1) & (np.abs(lat) > 0.2)
agree_steer = float((np.sign(steer[m]) == np.sign(lat[m])).mean())
riv = Anext[:, :, ci["RIV"]] > 0.5; bend = Anext[:, :, ci["SMDD"]] - Anext[:, :, ci["SMDV"]]; mb = riv & (np.abs(bend) > 0.1) & (np.abs(lat) > 0.2)
agree_bend = float((np.sign(bend[mb]) == -np.sign(lat[mb])).mean()) if mb.any() else float("nan")           # 등쪽(+) = 오른쪽 = 소스가 오른쪽(lat<0)일 때 정답
# 실현된 굽힘 펄스 부호(ADR-017 규칙의 실제 방향, +등쪽=오른쪽) vs 소스 쪽: 펄스가 있는 스텝만
OS = out["omega_sign"][:, 1:]; mo = (OS != 0) & (np.abs(lat) > 0.2)
agree_real = float((OS[mo] == -np.sign(lat[mo])).mean()) if mo.any() else float("nan"); n_dorsal = int((OS[mo] > 0).sum()); n_ventral = int((OS[mo] < 0).sum())
print(f"      실현 펄스 방향 = 소스 쪽 일치율 {agree_real:.2f} (n={mo.sum()}, 등쪽 {n_dorsal} / 배쪽 {n_ventral})")
print(f"{a.policy} | reach {out['reached'].mean():.3f} | H2-2 조향 부호 = 소스 쪽 일치율 {agree_steer:.2f} (n={m.sum()}, 판정 > 0.7, 우연 0.5) | 굽힘 방향 = 소스 쪽 일치율 {agree_bend:.2f} (n={mb.sum()}, 판정 > 0.6)")
json.dump({"reach": float(out["reached"].mean()), "agree_steer": agree_steer, "n_steer": int(m.sum()), "agree_bend": agree_bend, "n_bend": int(mb.sum()), "agree_realized": agree_real, "n_realized": int(mo.sum()), "n_dorsal": n_dorsal, "n_ventral": n_ventral}, open(os.path.join(os.path.dirname(a.policy), "weathervane.json"), "w"), indent=1)
