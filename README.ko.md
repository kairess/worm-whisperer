# worm-whisperer

**Grounding Language in a Connectome** — 자연어 명령을 *C. elegans* 커넥톰을 **통해** 행동으로 잇는 시뮬레이터. 언어 모델은 근육을 건드리지 않는다.

*English README: [README.md](README.md)* · **데모 페이지(시뮬레이션 녹화 재생, 미로·뉴런 판독): [kairess.github.io/worm-whisperer](https://kairess.github.io/worm-whisperer/)**

## 아이디어

"LLM 이 로봇을 조종하는" 데모는 대개 모델이 액추에이터를 직접 움직인다. 이 프로젝트는 반대다. OpenWorm c302 모델(302 뉴런, 발표된 커넥톰의 화학 시냅스와 갭정션)의 연결 구조는 **절대 바꾸지 않고**, 언어 모델은 운동뉴런이나 근육을 아예 자극할 수 없다. 모델이 하는 일은 **"가상 실험자"** 역할뿐이다. 실제 논문에 보고된 광유전학 / 감각 자극 프로토콜 중 하나를 골라 감각·명령·조향 개재뉴런의 짧은 화이트리스트에 전류를 주면, 그 아래 — 명령 선택, 운동 패턴, 2D 신체 — 는 전부 커넥톰 동역학과 신체 물리가 만든다.

```
"앞으로 가"     → AVB 명령 개재뉴런 자극 → 커넥톰 → 운동층 → 전진 0.18 mm/s, 0.56 Hz
"왼쪽으로 가"   → AVB + SMDV 머리 조향 뉴런 → 12초에 +52° 좌회전
"유턴해"        → AVA 후진 → RIV/SMDV 오메가턴 → AVB 전진 → +174° 재정향
"춤춰봐"        → 후진 + 오메가턴 + 머리 흔들기 반복 (피루엣, 12초에 +369°)
"유튜브 보는중" → RIS 수면활성뉴런 자극 → 운동 정지 (정지 비율 0.78)
```

작동하지 않는 것도 기록한다. 감각 자극(터치, 기피, 화학주성)은 방향을 고르지 못하는데, c302 세포 모델이 아틀라스의 억제 응답을 재현하지 못하기 때문이다. 또 c302 세포 모델로는 이동 리듬이 창발하지 않아 운동뉴런–근육 층은 문헌 모델(Boyle, Berri & Cohen 2012)로 대체했다. 아래 [되는 것과 안 되는 것](#되는-것과-안-되는-것) 참고.

## 동작 원리

```
 문장 ──► 문장 임베딩 ──► 프로토콜 혼합 헤드 ──► 자극 스케줄 (화이트리스트 뉴런만)
        (MiniLM, 고정)    (작은 MLP, 문장 267개)   예: AVB 5 pA, 10 s

 자극 ──► JAX 로 구현한 c302 C2 네트워크 (302 뉴런, 커넥톰 고정, 적합한 동역학 "Vfit")
      ──► AVB / AVA / SMD / RIV / RIS 막전위 판독
      ──► Boyle 2012 쌍안정 운동층 + 신장 수용기 (AVB/AVA 게이트, SMD 조향, RIV 오메가 굽힘)
      ──► 한천 위 2D 점탄성 막대 (저항력 이론) ──► 궤적, 곡률, 파동 주파수
```

모든 실험이 지키는 원칙 (`PLAN.md` 1.1절):

1. **커넥톰은 불변.** 적합하는 것은 동역학 가정(시냅스 문턱, 이득, 누설)이며 변형 이름(`V0`, `V1`, `Vfit`, …)으로 관리한다.
2. **LLM 은 조종사가 아니라 실험자.** `worm/llm/protocols.py` 의 프로토콜만 고를 수 있고, 채널은 감각·명령·조향 개재뉴런뿐이다. `validate()` 가 강제한다.
3. **모든 프로토콜에 근거 논문**, 커넥톰 밖의 모든 추가 기전(운동층, 조향, 오메가턴, RIS 정지)은 플래그로 분리된 명시적 가정이며 `docs/DECISIONS.md` 에 ADR 로 남긴다.
4. **부정 결과도 기록한다.**
5. 보고 수치는 dt 0.05 ms float64 로 재확인한다 (탐색은 dt 0.25).

## 되는 것과 안 되는 것

| 층 | 결과 | 문서 |
|---|---|---|
| c302 C2 의 JAX 재구현 | NEURON 대비 RMS 0.001 mV, CPU 에서 실시간 이상 | `docs/RESULTS_PHASE1.md` 1절 |
| Randi 2023 신호전파 아틀라스 비교 | 동역학 > 해부학 (검증 자극 140개에서 Spearman 0.13 vs 0.08, AUROC 0.65 vs 0.55). 예측력의 원천은 갭정션 확산, 화학 시냅스는 휴지에서 꺼져 있음 | 5–6절 |
| 억제 재현과 감각→명령 선택 | **부정.** 아틀라스 억제 쌍의 83% 는 커넥톰에 시냅스 경로가 없고, 억제를 켜면 예측력이 사라지며, 클래스별 파라미터 63개로도 회복 안 됨 | 7절 |
| c302 세포로 이동 | **부정.** 고유 진동 없음, 고유수용 피드백으로 파동 창발 없음, 채널 추가도 실패 | `docs/RESULTS_PHASE2.md` |
| 커넥톰 AVB/AVA 로 게이트한 Boyle 2012 운동층 | 전진 0.18 mm/s 0.56 Hz, 후진 0.11 mm/s (문헌 0.2 mm/s, 0.3–0.5 Hz) | `docs/RESULTS_PHASE3_4.md` 1절 |
| 조향, 유턴, 국소 탐색, RIS 정지 | 광유전학 등급 프로토콜 9개 모두 의도한 행동 | 1b절 |
| 감각 프로토콜 (터치, ASH 도망, 화학주성) | **부정.** 방향 선택 안 됨 (위 억제 문제와 같은 원인) | 2절, 5절 |
| 번역기 (임베딩 → 프로토콜 혼합) | 새 문장 정확도 0.86, 행동 수준에서 26/26 명령이 의도한 운동 축 도달 | 3–4절 |

## 빠른 시작

필요한 것: [uv](https://docs.astral.sh/uv/), Python 3.12 (uv 가 내려받음). `runs/` 의 기준 데이터 약 650 MB 는 git 에 없다 — [기준 데이터](#기준-데이터) 참고.

```bash
uv sync                                                  # 의존성 전부 (JAX, torch, NEURON, c302, sentence-transformers)
uv run python experiments/phase4_train_translator.py     # 번역기 학습 1회 (~1분, CPU)
uv run uvicorn worm.server.app:app --port 8000           # http://localhost:8000 에서 명령 입력
uv run pytest -q tests                                   # 테스트 19개, 약 25초
```

웹 UI 는 한국어와 영어를 받는다 ("앞으로 가", "turn left", "유턴해", "춤춰봐", "유튜브 보는중"). 벌레, 자극 중인 뉴런, 명령/조향 뉴런의 막전위, 번역기가 고른 프로토콜과 근거 논문을 보여준다.

### 기준 데이터

모든 실험은 c302 가 내보낸 NeuroML 네트워크 `runs/phase0/c302_C2_LW_Full_avb-ava/` 를 읽는다. 다른 컴퓨터의 `runs/` 를 복사해 오거나(그것으로 충분), 다시 생성한다. 재생성에는 Java(jNeuroML)와 C 컴파일러(NEURON `nrnivmodl`)가 필요하다.

```bash
# macOS:  brew install openjdk@21        Linux:  sudo apt install build-essential openjdk-21-jre-headless
uv run python experiments/phase0_c302_reference.py --full     # 몇 분, runs/phase0/ 생성
```

### 플랫폼별 참고

- **Linux + NVIDIA GPU.** `uv sync` 가 `jax[cuda12]` 0.7.1 과 CPU 전용 torch 를 설치한다. 배치 적합 스크립트(`phase1c_fit.py`, `phase1d_fit.py`, `phase1b_sigprop.py`)는 GPU 로 돈다: 적합 1스텝이 RTX 5090 에서 2초, 16코어 CPU 에서 43초. 단일 궤적 작업(테스트, 웹 UI)은 커널 지연이 지배해 CPU 가 3배 빠르므로 서버와 테스트는 `JAX_PLATFORMS=cpu` 로 고정한다. GPU 작업을 여러 개 동시에 돌리면 `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  - 의존성 고정 이유: jax 0.7.2+ 는 numpy ≥ 2 를 요구(프로젝트는 numpy < 2). PyPI 의 torch cu121 휠은 cuDNN 8.9 를 요구해 CUDA JAX(cuDNN ≥ 9.8)와 충돌하고, RTX 50 시리즈(sm_120)는 torch 2.2.2 가 지원하지 않으므로 `[tool.uv.sources]` 의 pytorch-cpu 인덱스에서 CPU 휠을 받는다. 임베딩은 `worm/llm/translator.py` 의 `torch_device()` 가 torch 빌드에 GPU 커널이 있을 때만 cuda 를 고른다.
- **macOS Intel.** 조건부 의존성으로 JAX 0.4.38 과 torch 2.2.2(Intel 휠이 있는 마지막 버전)를 유지한다. Phase 0 에는 `brew install openjdk@21` 이 필요하고, NEURON 기전 컴파일 시 CommandLineTools 의 깨진 libc++ 헤더는 스크립트가 `CXX="clang++ -I$(xcrun --show-sdk-path)/usr/include/c++/v1"` 로 우회한다. `pynml -neuron -run` 은 pip 설치 NEURON 을 못 찾으므로 스크립트가 내보내기만 하고 `nrnivmodl` 로 직접 컴파일해 `LEMS_*_nrn.py` 를 실행한다.
- **Apple Silicon.** 미검증. `pyproject.toml` 의 고정을 풀어도 된다.
- Python 은 3.12 까지 (`requires-python <3.13`: torch 2.2.2 에 3.13 휠이 없음). uv 가 3.13 을 고르면 `uv sync -p 3.12`. `[dependency-groups]`(pytest)는 uv ≥ 0.4.27 에서만 설치되므로 오래된 uv 면 `uv self update`.

## 저장소 구조

```
worm/neural/     connectome.py (NeuroML 파서), jaxsim.py (C2 시뮬레이터), variants.py (V0/V1/Vfit, θ), readers/
worm/body/       rod2d.py (2D 막대), boyle_motor.py (Boyle 2012 운동층), muscle_map.py
worm/sim.py      신경–신체 결합 (Worm), 게이트, 조향, 오메가 굽힘, 행동 기술자
worm/llm/        protocols.py (화이트리스트), phrases.py (학습 문장), translator.py
worm/server/     FastAPI + WebSocket 서버;  web/index.html  UI
experiments/     실험별 스크립트, phase0 … phase4
tests/           테스트 19개: NEURON 동등성, 신체 모델, 운동층, 프로토콜, 통합
docs/            결과, 결정, 참고문헌
runs/            산출물, git 제외 (약 650 MB)
history/         Claude Code 세션 원본 기록 (jsonl)
```

## 문서

| 문서 | 내용 |
|------|------|
| [PLAN.md](PLAN.md) | 목표, 원칙, 아키텍처, 단계별 로드맵, 리스크 |
| [docs/COMMANDS.md](docs/COMMANDS.md) | 명령어 ↔ 뉴런 자극 매핑표와 근거 논문 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 설계 결정 기록 ADR-001 … ADR-016 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Phase 0 기준 실행과 C2 모델 사양 |
| [docs/RESULTS_PHASE1.md](docs/RESULTS_PHASE1.md) | JAX 재구현 검증, 쌍안정성, 신호전파 아틀라스 비교, 파라미터 적합, 억제(부정 결과) |
| [docs/BODY_MODEL.md](docs/BODY_MODEL.md) | 2D 신체 모델 사양과 검증 |
| [docs/RESULTS_PHASE2.md](docs/RESULTS_PHASE2.md) | 신경–신체 결합, 고유수용 피드백, 진동자 탐색(부정 결과) |
| [docs/RESULTS_PHASE3_4.md](docs/RESULTS_PHASE3_4.md) | 프로토콜, 조향/유턴/정지, 번역기, UI, 화학주성(부정 결과) |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 참고 문헌, 도구, 데이터셋 |
| [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | 논문 개요 초안 |
| [docs/SESSION_LOG.md](docs/SESSION_LOG.md) | 작업 기록과 다른 컴퓨터에서 이어가기 안내. 원본 대화는 `history/*.jsonl` |
| [CLAUDE.md](CLAUDE.md) | Claude Code 가 자동으로 읽는 프로젝트 지침 |

## 핵심 참고문헌

- Gleeson et al. 2018, *c302: a multiscale framework for modelling the nervous system of C. elegans* (여기서 쓴 뉴런 모델과 커넥톰).
- Randi et al. 2023, *Neural signal propagation atlas of C. elegans*, Nature (동역학을 비교한 기능 데이터).
- Boyle, Berri & Cohen 2012, *Gait modulation in C. elegans: an integrated neuromechanical model*, Front. Comput. Neurosci. (운동층).
- Chalfie et al. 1985; Gray, Hill & Bargmann 2005; Turek et al. 2016 (터치 회로, 조향/오메가턴, RIS 정지 프로토콜).

전체 목록은 [docs/REFERENCES.md](docs/REFERENCES.md).

## 상태

첫 번째 프로젝트 마일스톤을 2026-09-03 에 마감했다. Claude Code 와 함께 개발했으며 세션 원본 전체가 `history/` 에 있다. 남은 과제는 [docs/SESSION_LOG.md](docs/SESSION_LOG.md) 4절.
