# worm-whisperer

**고정된 커넥톰으로 무엇이 되고 안 되는가** — 학습 에이전트가 화이트리스트 개재뉴런만 자극할 수 있는 전신 *C. elegans* 시뮬레이터. 302뉴런 배선, 운동층, 몸이 나머지를 만든다.

![T-미로, 먹이가 오른쪽: 왼쪽 팔에 먼저 들어갔다가 냄새가 옅어지면 후진해 먹이에 도달 (시뮬레이션 녹화)](docs/assets/tmaze_right.gif)

*English README: [README.md](README.md)*

**데모 페이지(시뮬레이션 녹화 재생, 미로·뉴런 판독): [kairess.github.io/worm-whisperer](https://kairess.github.io/worm-whisperer/)**

## 질문

OpenWorm c302 커넥톰(302뉴런, 발표된 배선의 화학 시냅스와 갭정션)을 연결 하나 바꾸지 않고 두고 묻는다. **이 배선으로 어떤 행동이 만들어지고, 어떤 행동은 안 되는가?** 에이전트(진화 전략 정책)는 근육을 움직이지 않는다. 광유전학 실험자처럼 감각·명령·조향 개재뉴런의 짧은 화이트리스트에만 전류를 준다. 그 아래는 전부 고정이다: 커넥톰 동역학, 문헌 운동층(Boyle, Berri & Cohen 2012), 벽이 있는 한천 위 2D 점탄성 몸.

모든 가설은 학습 전에 판정 기준과 함께 사전 등록했고(`docs/PLAN_WORMGYM.md`), 부정 결과도 그대로 남겼다.

## 주요 결과 (사전 등록, 무작위 시작 128–256회)

| 질문 | 판정 | 핵심 수치 |
|---|---|---|
| 명령·조향 개재뉴런(AVB, AVA, SMDD, SMDV, RIV)만 자극해 화학주성이 되는가? | ✓ | 2.5 mm, 40초 안 도달률 0.914 (무작위 0.04) |
| 감각 뉴런(ASE, AWC, AWA, ASK, ASH, PLM, ALM, AVM)만 자극하면? | ✗ | 0.109 ≈ 우연 0.125 — 이 모델에서 감각→명령 단계는 끊겨 있다(정적 자극에서도 동일) |
| 학습이 피루엣 규칙을 스스로 찾는가? | ✓ (규칙) / △ (충분성) | 농도 하강 → 후진+오메가, 상승 → 전진 (3 시드 중 2 에서 ρ −1.00); 후진 개시율은 Pierce-Shimomura 1999 처럼 dC/dt 에 따라 단조 감소. 후진+오메가만으로는 2 시드 실패(0.05–0.11), 1 시드는 굽힘 상시 전략으로 2.5 mm 0.328 (전체 회로 0.914) |
| 최소 회로는? | | 후진(AVA) + 깊은 굽힘(RIV) + 조향(SMD): 학습된 정책에서 하나만 절제해도 0.07–0.27; RIV 없이 재학습하면 실패(0.19), SMD 없이는 최대 0.33 |
| 정지에 수면 뉴런(RIS)이 필요한가? | ✗ | 쓰지 않음; 명령 드라이브를 낮춰 멈춘다 |
| 명령 뉴런을 직접 자극해도 실제 배선이 중요한가? | ✓ | 배선을 뒤섞은 커넥톰(가중치·개수·부호 보존): 재학습 0.141, 전이 0.023 vs 실제 배선 0.871 — 뒤섞인 배선에서는 깊은 굽힘 펄스가 거의 발사되지 않는다 |
| 몸 폭 4배 통로에서 모퉁이는 어떻게 도는가? | | 벽에 눌린 깊은 굽힘으로만; 머리 조향만으로는 0/32. **굽힘의 등/배 방향이 곧 회전 방향** |
| 배쪽 굽힘만 있으면(기본 오메가턴)? | | T-미로 첫 진입 128/128 왼쪽 팔; 오른쪽 통로 0/32; 3×3 격자 불가 |
| 실제 벌레처럼 SMDD/SMDV 가 굽힘 방향을 정하게 하면? | ✓ (ADR-017) | 오른쪽 통로를 배운다(1.00) — 단 정책은 한손잡이가 된다; 통로 폭 0.25–0.35 mm 에서 강건 |
| T 갈림길에서 맞는 팔을 고를 수 있는가? | | 시간 감각만으로는 원리상 불가(갈림길에서 두 팔의 냄새가 같음) → 학습된 전략은 "왼쪽 먼저, 나빠지면 후진" (도달 1.00, 오른쪽 목표는 7–38초 손해). 좌우 농도차를 주면 굽힘 방향이 소스 쪽을 따른다(펄스의 73 %; 없으면 62 %) |
| 그런데 왜 정확히 겨눈 오른쪽 진입도 중단되는가? | | 벽에 막힌 정체가 "농도가 안 오른다" 로 읽혀 후진+배쪽 굽힘 반사가 발동 — 이 반사가 모퉁이를 도는 유일한 수단이기도 하다. 반사를 끄면 모퉁이도 못 돈다 |

**실제 벌레에 대한 검증 가능한 예측.** 좁은 T-미로에서 팔 선택은 교차점 통과 순간의 깊은 굽힘 방향(등/배)이 결정한다. 등쪽 회전을 억제하는 조건에서는 배쪽 팔 편향과 반대 목표의 도달 지연이 나타난다. 벽 정체는 피루엣을 촉발한다. 실제 벌레는 선천적 좌우 편향이 없고(Gourgou 2021, DI 0.03) 기울기에 따라 등/배 회전을 고르므로(Nature Neurosci 2026), 모델에서는 등쪽 굽힘과 좌우 감각이 둘 다 있어야 그것이 재현된다.

**정직한 한계.** 운동층·조향·깊은 굽힘 규칙은 커넥톰 밖의 문헌 기반 가정이다(ADR-012/013/016/017). 몸에 벽 밀기 추진(Park 2008)이 없어 통로에서 느리다. 등/배 비용 비대칭이 없어 실제 배쪽 선호를 설명하지 못한다. 학습 알고리즘 하나(ES), 작은 정책 하나. 3×3 격자의 강건한 해는 없다.

## 동작 원리

```
 관측 ──► 정책 (MLP, 은닉 16) ──► 화이트리스트 뉴런 자극 진폭 (0–6 pA, 0.5 s 마다)
 (로그 농도 변화, 벽 접촉, …)        AVB AVA SMDD SMDV RIV  (운동뉴런·근육은 절대 아님)

 자극 ──► JAX 로 구현한 c302 C2 네트워크 (302 뉴런, 배선 고정, 적합한 동역학 "Vfit")
      ──► AVB / AVA / SMD / RIV / RIS 막전위 판독
      ──► Boyle 2012 쌍안정 운동층: 전진/후진 게이트, 머리 조향, 깊은 굽힘 펄스
      ──► 벽이 있는 한천 위 2D 점탄성 막대 ──► 궤적, 냄새, 보상
```

학습은 GPU 배치 시뮬레이터(`worm/env/batch.py`, 롤아웃당 수백 마리, B=256 에서 초당 약 6 에피소드) 위의 OpenAI-ES. 평가는 항상 사전 등록된 규격으로 새 무작위 시작 128–256회.

## 빠른 시작

필요한 것: [uv](https://docs.astral.sh/uv/), Python 3.12(uv 가 내려받음), 선택적으로 NVIDIA GPU. c302 기준 네트워크(`runs/phase0/…`, 약 650 MB, git 제외)가 모든 실험에 필요하다 — 다른 컴퓨터의 `runs/` 를 복사하거나 재생성(Java·C 컴파일러 필요, `docs/SESSION_LOG.md` 5절).

```bash
uv sync
uv run pytest -q tests                                   # 테스트 21개, 약 30초 (CPU)

# 명령·조향 채널 화학주성 (커리큘럼: 1.5 mm → 2.5 mm)
uv run python experiments/wormgym_es.py --src_dist 1.5 --pop 32 --ep_per 8 --gens 40 --out runs/wormgym/near
uv run python experiments/wormgym_es.py --src_dist 2.5 --init runs/wormgym/near/theta.npy --gens 100 --out runs/wormgym/far
uv run python experiments/wormgym_es.py --eval runs/wormgym/far/theta.npy --episodes 256      # 도달률, n=256

# 미로 (벽, 접촉 관측, 등/배 굽힘 규칙)
uv run python experiments/wormgym_es.py --maze tmaze --touch --omega_smd_dir --episode_s 120 --init runs/wormgym/far/theta.npy --gens 40 --out runs/wormgym/tmaze
uv run python experiments/wormgym_maze_analyze.py runs/wormgym/tmaze/theta.npy --maze tmaze --touch --omega_smd_dir
```

### 플랫폼별 참고
- **Linux + NVIDIA GPU.** `uv sync` 가 `jax[cuda12]` 0.7.1 과 CPU 전용 torch 를 설치한다. 배치 학습·평가는 GPU, 서버·테스트·단일 궤적은 CPU 가 빠르다(`JAX_PLATFORMS=cpu` 기본). GPU 작업을 여러 개 동시에 돌리면 `XLA_PYTHON_CLIENT_PREALLOCATE=false`. CPU 멀티프로세스 학습(`wormgym_h1.py`)은 워커당 스레드 1개로 제한돼 있다.
- **macOS Intel.** 조건부 의존성으로 JAX 0.4.38 / torch 2.2.2 유지. Phase 0 재생성에는 `brew install openjdk@21` 과 NEURON 컴파일 우회(스크립트에 반영)가 필요.
- Python 은 3.12 까지(torch 2.2.2 의 3.13 휠 없음). 오래된 uv 면 `uv self update`.

## 1부 (이전 작업): 자연어 명령을 내리는 가상 실험자

학습 에이전트 이전에는 같은 시뮬레이터를 작은 번역기로 움직였다. 문장 임베딩(MiniLM) → 문헌 근거가 있는 프로토콜 13개의 혼합 → 화이트리스트 전류.

```
"앞으로 가"     → AVB 명령 개재뉴런         → 전진 0.18 mm/s, 0.56 Hz
"왼쪽으로 가"   → AVB + SMDV                → 12초에 +52°
"유턴해"        → AVA → RIV/SMDV 오메가턴 → AVB → +174° 재정향
"유튜브 보는중" → RIS 수면활성뉴런           → 운동 정지 (정지 비율 0.78)
```

새 문장 정확도 0.86, 26/26 명령이 의도한 운동 축 도달. 감각 프로토콜(터치, 기피, 화학주성)은 위에서 밝힌 이유로 작동하지 않는다.

```bash
uv run python experiments/phase4_train_translator.py
uv run uvicorn worm.server.app:app --port 8000        # http://localhost:8000
```

자세한 내용: `docs/RESULTS_PHASE3_4.md`, `docs/COMMANDS.md`.

## 저장소 구조

```
worm/neural/     connectome.py (NeuroML 파서), jaxsim.py (C2 시뮬레이터), variants.py (V0/V1/Vfit, θ, 배선 뒤섞기 대조군)
worm/body/       rod2d.py (2D 막대 + 벽), boyle_motor.py, motor_block.py (JAX 블록), muscle_map.py
worm/sim.py      신경–신체 결합 (Worm), 게이트, 조향, 깊은 굽힘 펄스
worm/env/        gym.py (numpy 환경), batch.py (GPU 배치 환경), mazes.py, chem.py
worm/llm/        protocols.py (화이트리스트), phrases.py, translator.py      worm/server/  웹 UI
experiments/     wormgym_*.py (ES, 분석, 오라클, 데모 내보내기); 1부의 phase0–4 스크립트
tests/           테스트 21개    docs/  결과·결정(ADR-001…017)·계획    docs/index.html  데모 페이지
runs/            산출물, git 제외
```

## 문서

| 문서 | 내용 |
|------|------|
| [docs/PLAN_WORMGYM.md](docs/PLAN_WORMGYM.md) | Worm Gym: 사전 등록 가설 H1–H5-9, 판정 전부, 기전, 문헌 대조 |
| [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | 논문 주장·그림·투고 전 실험 E1–E5 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | ADR-001 … ADR-017 |
| [docs/RESULTS_PHASE1.md](docs/RESULTS_PHASE1.md) | JAX 재구현, 신호전파 아틀라스, 억제(부정 결과) |
| [docs/RESULTS_PHASE2.md](docs/RESULTS_PHASE2.md) | 신체 결합, 진동자 탐색(부정 결과) |
| [docs/RESULTS_PHASE3_4.md](docs/RESULTS_PHASE3_4.md) | 프로토콜, 조향/유턴, 번역기, UI |
| [docs/SESSION_LOG.md](docs/SESSION_LOG.md) | 작업 기록과 다른 컴퓨터에서 이어가기 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 참고 문헌, 도구, 데이터셋 |

## 핵심 참고문헌

- Gleeson et al. 2018, *c302* (뉴런 모델과 커넥톰). Randi et al. 2023, Nature (신호전파 아틀라스).
- Boyle, Berri & Cohen 2012 (운동층). Chalfie 1985; Gray, Hill & Bargmann 2005; Turek 2016 (프로토콜).
- Pierce-Shimomura, Morse & Lockery 1999 (피루엣). Gourgou et al. 2021, iScience (T-미로). *Neural sequences underlying directed turning in C. elegans*, Nature Neurosci 2026 (등/배 회전 제어). Park et al. 2008, PLoS ONE (구조 환경 보행).

## 상태

1부 2026-09-03 마감; Worm Gym 과 미로 항법 2026-09-03…06. 투고 전 실험 E1–E5(교차점 기전, 시드 반복, 폭 민감도, 문헌 대조, 배선 뒤섞기 대조군)를 마쳤다 (`docs/PAPER_OUTLINE.md` 3·6절). Claude Code 와 함께 개발했고 세션 원본은 `history/`.
