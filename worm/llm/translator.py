"""옵션 B-1: 고정 다국어 임베딩(paraphrase-multilingual-MiniLM-L12-v2, 384-d) → 자극 패턴 u ∈ R^{K×T} 회귀 (behavior cloning).
- 출력 공간은 화이트리스트 채널 K(=len(CHANNELS)) × 시간 구간 T(0.5 s 단위, 12 s). 운동뉴런·근육은 출력에 없다(구조적 제약).
- 교사 신호: protocols.py 의 스케줄을 (K, T) 행렬로 이산화. 손실 MSE + L1 희소성.
- 디코드: u → 스케줄 (채널별 구간 진폭, 0.5 pA 미만 절사, 최대 6 pA 클리핑) → 화이트리스트 검증.
"""
from __future__ import annotations
import os, json, numpy as np
from .protocols import CHANNELS, PROTOCOLS, validate
CH = list(CHANNELS.keys()); K = len(CH); T_BIN = 0.5; T = 24; AMP_MAX = 6.0
_models = {}
EMBED_MODELS = {"minilm": "paraphrase-multilingual-MiniLM-L12-v2", "e5": "intfloat/multilingual-e5-small"}
def torch_device():
    """설치된 torch 빌드가 실제로 커널을 가진 GPU 일 때만 cuda. (예: torch 2.2.2 는 RTX 5090 sm_120 미지원 → cpu)"""
    import torch
    if not torch.cuda.is_available(): return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    return "cuda" if f"sm_{major}{minor}" in torch.cuda.get_arch_list() else "cpu"

def embed(texts, model="minilm"):
    if model not in _models:
        from sentence_transformers import SentenceTransformer
        _models[model] = SentenceTransformer(EMBED_MODELS[model], device=torch_device())
    texts = [("query: " + t) if model == "e5" else t for t in texts]
    return np.asarray(_models[model].encode(list(texts), normalize_embeddings=True), np.float32)

def schedule_to_u(schedule):
    u = np.zeros((K, T), np.float32); inv = {n: k for k, c in enumerate(CH) for n in CHANNELS[c]}
    for t0, t1, stim in schedule:
        b0, b1 = int(t0 / T_BIN), min(T, int(np.ceil(t1 / T_BIN)))
        for n, a in stim.items(): u[inv[n], b0:b1] = a
    return u

def u_to_schedule(u, thresh=0.5):
    u = np.clip(np.asarray(u).reshape(K, T), 0, AMP_MAX); sch = []
    for k in range(K):
        for b in range(T):
            if u[k, b] >= thresh: sch.append((b * T_BIN, (b + 1) * T_BIN, {n: float(u[k, b]) for n in CHANNELS[CH[k]]}))
    return validate(sch)

class Translator:
    """임베딩 → (프로토콜 혼합 가중치 p ∈ Δ^C, 진폭 배율 a > 0) → u = a · Σ_c p_c · u_c.
    출력은 항상 화이트리스트 프로토콜 패턴의 볼록 결합이라 구조적으로 커넥톰 원칙을 지키고, 혼합은 합성 명령("뒤로 천천히")을 표현한다."""
    def __init__(self, protos_u: dict, dim=384, hidden=256, seed=0, dropout=0.2):
        import torch, torch.nn as nn
        torch.manual_seed(seed); self.names = list(protos_u.keys()); C = len(self.names)
        self.U = torch.tensor(np.stack([protos_u[n] for n in self.names]).reshape(C, -1))       # (C, K*T)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, C + 1))
    def _forward(self, X):
        import torch
        out = self.net(X); logits, a = out[:, :-1], torch.nn.functional.softplus(out[:, -1:]) + 0.5
        p = torch.softmax(logits, 1); u = a * (p @ self.U); return logits, p, a, u
    def fit(self, X, y, epochs=600, lr=3e-3, verbose=False):
        import torch
        X = torch.tensor(X); y = torch.tensor(y); Ut = self.U[y]; opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-3)
        for ep in range(epochs):
            self.net.train(); logits, p, a, u = self._forward(X)
            loss = torch.nn.functional.cross_entropy(logits, y) + 0.05 * ((u - Ut) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and ep % 200 == 0: print(f"  epoch {ep} loss {loss.item():.4f}")
        return self
    def predict(self, X, gamma=2.0):
        """gamma > 1: 디코딩 시 혼합 확률을 p^γ 로 선명화 (ADR-014 보완). 확신이 낮은 혼합(예: p 0.45)이 펄스 진폭을 절반으로 희석해
        문턱형 기전(오메가턴 RIV 30 mV, ADR-016)을 못 넘는 것을 막는다. 진짜 혼합(두 프로토콜이 비슷한 p)은 비율이 유지된다. gamma=1 이면 원래 혼합."""
        import torch
        self.net.eval()
        with torch.no_grad():
            logits, p, a, u = self._forward(torch.tensor(X))
            if gamma != 1.0:
                p = p ** gamma; p = p / p.sum(1, keepdim=True); u = a * (p @ self.U)
            return u.numpy().reshape(len(X), K, T), p.numpy(), a.numpy()[:, 0]
    def classify(self, X):
        u, p, a = self.predict(X); return [self.names[i] for i in p.argmax(1)], p
    def save(self, path):
        import torch; torch.save(self.net.state_dict(), path)
    def load(self, path):
        import torch; self.net.load_state_dict(torch.load(path)); return self

def nearest_protocol(u, protos_u):
    """예측 u 와 각 프로토콜 교사 u 의 코사인 유사도로 가장 가까운 프로토콜."""
    v = u.reshape(-1); best, bs = None, -2
    for name, pu in protos_u.items():
        p = pu.reshape(-1); sim = float(v @ p / (np.linalg.norm(v) * np.linalg.norm(p) + 1e-9))
        if sim > bs: best, bs = name, sim
    return best, bs
