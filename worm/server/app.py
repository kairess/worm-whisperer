"""Phase 5: FastAPI + WebSocket 서버. 브라우저 채팅 명령 → 번역기 → 화이트리스트 스케줄 → Worm 시뮬레이션 → 프레임 스트리밍.
실행: uv run uvicorn worm.server.app:app --port 8000  → http://localhost:8000
"""
import os, sys, json, asyncio, time, numpy as np
sys.path.insert(0, os.getcwd())
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 대화형 단일 궤적 시뮬레이션은 CPU 가 GPU 보다 빠르다 (README Linux 절). GPU 를 쓰려면 JAX_PLATFORMS=cuda
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from worm.sim import Worm
from worm.llm.protocols import PROTOCOLS
from worm.llm import translator as T

NML = os.environ.get("WORM_NML", "runs/phase0/c302_C2_LW_Full_avb-ava/c302_C2_LW_Full_avb-ava.net.nml")
LABELS = {"forward": "전진 중", "reverse": "후진 중", "stop": "정지", "turn_left": "왼쪽으로 회전 중", "turn_right": "오른쪽으로 회전 중",
          "omega_turn": "유턴 중", "local_search": "춤추는 중 (국소 탐색)", "head_sweep": "두리번거리는 중", "quiescence_RIS": "유튜브 보는중 (RIS 정지)",
          "quiescence_ALA": "피곤해서 쉬는 중 (ALA)", "escape": "도망가는 중", "forward_touch": "꼬리 터치 반응", "reverse_touch": "머리 터치 반응"}
app = FastAPI()
_state = {"worm": None, "translator": None, "protos_u": None, "meta": None, "schedule": [], "t0": 0.0, "label": "정지", "rationale": "", "mix": {}}

def _load():
    if _state["worm"] is None:
        _state["worm"] = Worm(NML, "Vfit", motor="boyle")
        protos_u = {n: T.schedule_to_u(fn()) for n, (g, fn, r) in PROTOCOLS.items()}; _state["protos_u"] = protos_u
        meta = json.load(open("runs/phase4/translator.json")); _state["meta"] = meta
        _state["translator"] = T.Translator(protos_u, dim=meta["dim"]).load("runs/phase4/translator.pt")
    return _state["worm"]

def command(text):
    tr, meta = _state["translator"], _state["meta"]
    u, p, a = tr.predict(T.embed([text], meta["embedding"])); names = tr.names
    top = int(p[0].argmax()); name = names[top]
    sch = T.u_to_schedule(u[0]); w = _state["worm"]
    _state["schedule"] = [(w.t / 1000 + t0, w.t / 1000 + t1, stim) for t0, t1, stim in sch]
    _state["label"] = LABELS.get(name, name); _state["rationale"] = f"{name} (p={p[0][top]:.2f}): {PROTOCOLS[name][2]}"
    _state["mix"] = {names[i]: float(p[0][i]) for i in np.argsort(-p[0])[:3]}
    chans = sorted({n for _, _, st in sch for n in st}); return {"protocol": name, "label": _state["label"], "rationale": _state["rationale"], "mix": _state["mix"], "neurons": chans, "duration_s": max([t1 for _, t1, _ in sch] + [0])}

@app.get("/")
async def index():
    return HTMLResponse(open(os.path.join(os.path.dirname(__file__), "..", "..", "web", "index.html"), encoding="utf-8").read())

@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept(); w = _load(); loop = asyncio.get_event_loop()
    async def sim_loop():
        while True:
            t = w.t / 1000; I = {}
            for t0, t1, stim in _state["schedule"]:
                if t0 <= t < t1:
                    for k, v in stim.items(): I[k] = I.get(k, 0.0) + v
            rec = await loop.run_in_executor(None, w.step, I)
            V = np.asarray(w.neural.S.V); isM = w.net.is_muscle()
            frame = {"t": t, "x": rec["x"].tolist(), "kappa": rec["kappa"].tolist(), "A_D": rec["A_D"].tolist(), "A_V": rec["A_V"].tolist(),
                     "gates": list(rec["gates"]), "readout": rec["readout"], "label": _state["label"], "rationale": _state["rationale"], "mix": _state["mix"],
                     "V": [round(float(v), 1) for v in V[~isM]], "names": None, "stim": {k: v for k, v in I.items()}}
            await sock.send_text(json.dumps(frame)); await asyncio.sleep(0.005)
    task = asyncio.create_task(sim_loop())
    try:
        names = [n for n, m in zip(w.net.names, w.net.is_muscle()) if not m]; await sock.send_text(json.dumps({"names": names}))
        while True:
            msg = await sock.receive_text(); data = json.loads(msg)
            if "command" in data: await sock.send_text(json.dumps({"ack": command(data["command"])}))
    finally:
        task.cancel()
