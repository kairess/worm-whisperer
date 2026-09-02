"""Phase 4 (B-1): 문장 → 프로토콜 혼합 → 자극 패턴 번역기. 임베딩 모델 2종 비교, 문장 단위 분할(프로토콜당 5문장 검증), 5 시드 평균."""
import os, sys, json, numpy as np; sys.path.insert(0, os.getcwd())
from worm.llm.phrases import PHRASES
from worm.llm.protocols import PROTOCOLS
from worm.llm.translator import embed, schedule_to_u, u_to_schedule, Translator, CH, K, T
protos_u = {n: schedule_to_u(fn()) for n, (g, fn, r) in PROTOCOLS.items()}; names = list(protos_u)
os.makedirs("runs/phase4", exist_ok=True); results = {}
for model in ["minilm", "e5"]:
    accs = []; last = None
    for seed in range(5):
        rng = np.random.default_rng(seed); train, test = [], []
        for name, ph in PHRASES.items():
            idx = rng.permutation(len(ph)); test += [(ph[i], name) for i in idx[:5]]; train += [(ph[i], name) for i in idx[5:]]
        Xtr = embed([p for p, _ in train], model); Xte = embed([p for p, _ in test], model)
        ytr = np.array([names.index(n) for _, n in train]); yte = np.array([names.index(n) for _, n in test])
        tr = Translator(protos_u, dim=Xtr.shape[1], seed=seed).fit(Xtr, ytr)
        pred, p = tr.classify(Xte); acc = float(np.mean([pr == n for pr, (_, n) in zip(pred, test)])); accs.append(acc); last = (tr, test, pred, p)
    results[model] = accs; print(f"{model:7s} held-out accuracy over 5 seeds: {np.mean(accs):.3f} ± {np.std(accs):.3f}  {np.round(accs, 2).tolist()}")
best = max(results, key=lambda m: np.mean(results[m])); print("best embedding:", best)
# 최종 모델: 전체 문장으로 학습해 저장 (배포용), 검증 수치는 위 교차 분할 값을 보고
allp = [(p, n) for n, ph in PHRASES.items() for p in ph]; X = embed([p for p, _ in allp], best); y = np.array([names.index(n) for _, n in allp])
tr = Translator(protos_u, dim=X.shape[1], seed=0).fit(X, y); tr.save("runs/phase4/translator.pt"); json.dump({"embedding": best, "names": names, "dim": int(X.shape[1])}, open("runs/phase4/translator.json", "w"))
novel = ["살살 앞으로 가줘", "왼쪽으로 살짝 틀어", "졸려", "빨리 도망가", "뒤로 천천히", "제자리에서 신나게 흔들어", "유튜브 보는중", "오른쪽으로 가면서 춤춰", "코를 살짝 건드렸어"]
u, p, a = tr.predict(embed(novel, best))
for ph, pp, aa, uu in zip(novel, p, a, u):
    top = np.argsort(-pp)[:2]; print(f"   '{ph}' → {names[top[0]]} {pp[top[0]]:.2f}, {names[top[1]]} {pp[top[1]]:.2f}; amp×{aa:.2f}, max {uu.max():.1f} pA, 채널 수 {(uu.max(1) > 0.5).sum()}")
    u_to_schedule(uu)
json.dump(results, open("runs/phase4/embedding_comparison.json", "w"))
