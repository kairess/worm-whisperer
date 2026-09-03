# 작업 기록 (대화 요약) — 2026-09-02 ~ 09-03

원본 대화 전체는 `history/*.jsonl` (Claude Code 세션 기록, 도구 호출과 출력 포함). 이 문서는 사람이 읽기 위한 요약이다.

## 1. 출발점과 목표
- 요청: OpenWorm으로 LLM 명령("왼쪽으로 가", "춤춰봐", "유튜브 보는중")에 반응하는 꼬마선충 시뮬레이터. 뉴런 구조를 유지하고 과학적으로 의미 있게.
- 확정된 원칙 (PLAN.md 1.1): 커넥톰 불변, LLM은 조종사가 아니라 실험자(화이트리스트 프로토콜만), 모든 프로토콜에 근거 논문, 재현 가능, 문헌값과 비교.
- 방향 결정: API 방식 대신 **학습형 번역기(옵션 B)** — 로컬 LLM 임베딩 → 뉴런 자극 패턴. 출력 채널을 감각·명령 뉴런으로 제한.
- 환경: uv, Intel Mac(JAX 0.4.38 고정), 2D 신체 모델로 확정, Sibernetic 제외.

## 2. 진행 순서와 핵심 발견 (시간순)
1. **Phase 0** c302 C2 를 NEURON으로 실행. FW 예제의 파동은 근육 직접 자극이었음. 부분회로는 폭주, 전체 네트워크는 안정.
2. **Phase 1** NeuroML 파서 + JAX 시뮬레이터. NEURON 대비 뉴런 RMS 0.001 mV. dt 0.25에서 실시간 이상.
3. 지속 자극 ≥2 pA에서 전역 상향 상태(쌍안정). AVB는 예외.
4. **Phase 1b** Randi 2023 신호전파 아틀라스 비교: 동역학 모델 > 해부학(유의), 예측력은 갭정션 확산, 화학 시냅스는 휴지에서 꺼짐(Vth 0 mV), 억제 재현율 0. 상관은 측정 상한(ρ 0.14) 근접.
5. **Phase 1c** 전역 파라미터 6개 적합 → `Vfit`. 검증 AUROC 0.646, 폭주 없음, 터치 회로 논리 정성 재현. 최적화는 억제를 제거.
6. **Phase 2** 2D 막대 신체(0.24 mm/s 검증). c302 세포로는 이동 불가: 고유수용 피드백 두 방식, 고립 세포 진동 없음, 채널 확장(KCa, 느린 K) 모두 실패. Gao 2018 진동은 50 s 주기.
7. **Phase 2c** Boyle 2012 운동층 채택(ADR-012). 전진 0.18 mm/s 0.5 Hz 창발. Vfit 의 AVB/AVA 로 게이트 → 명령 자극으로 전진/후진.
8. **Phase 3** 프로토콜 13개, 조향(SMD → 머리 곡률 + 수용기 기준 이동, ADR-013), RIS 정지 게이트. 감각 프로토콜·화학주성·ALA 실패.
9. **Phase 4** 번역기(MiniLM 임베딩 → 프로토콜 혼합, ADR-014): 새 표현 0.84, 행동 축 일치 0.96. (09-03 개정 후 0.86 / 1.00)
10. **Phase 5** FastAPI+WebSocket 웹 UI.
11. fit_D(억제·터치 방향 제약 추가) 실패 → 전역 파라미터로는 불가.
12. 사용자 피드백: 좌/우 반대 → 규약 수정(배쪽 SMDV = 왼쪽). 코 터치 시 전진은 모델 한계(억제 부재)로 설명.

## 3. 현재 상태
- 실행: `uv sync` 후 `uv run python experiments/phase0_c302_reference.py --full` (기준 네트워크 생성, NEURON/Java 필요) → `uv run python experiments/phase4_train_translator.py` → `uv run uvicorn worm.server.app:app --port 8000`.
- 테스트: `uv run pytest -q tests` (18개, 약 2분).
- `runs/` 는 git 제외(약 700 MB). 다른 컴퓨터에서는 Phase 0 스크립트로 `runs/phase0/c302_C2_LW_Full_avb-ava/` 를 재생성해야 모든 실험이 돈다. 번역기 가중치(`runs/phase4/translator.pt`)는 재학습 1분.

## 3b. 2026-09-03 (두 번째 Mac 세션): Phase 1d 시작 — 억제 재현과 감각→명령 선택
- 구조 분석: 아틀라스 억제 쌍 150개 중 커넥톰 시냅스 경로가 있는 것은 26개뿐(직접 억제 3개). 핵심 명령 경로(AVD→AVA, PVC→AVB, AVM→AVB)는 CeNGEN 극성이 "상충"이라 V1-split 이 절반을 억제로 쪼갬. → `RESULTS_PHASE1.md` 7.1
- θ8(억제 동작점 분리) 13개 탐색: 억제 재현율은 기저율만큼만 오르고 예측력 하락. 부정 결과 → 7.2
- ADR-015: 연결 클래스별 파라미터 θ63 (세포 유형 S/I/M 쌍 × 부호). `connection_classes`, `apply_theta_class`, NeuroML 세포 유형 파싱(`Network.ntype`), `phase1d_fit.py`, `phase1d_inh_scan.py`, `--theta_class` 옵션(phase1b_sigprop, phase1_circuits, Worm).
- fit_E → fit_F: Chalfie 회로 방향 달성(PLM +2.3, ALM +2.4 mV) + 학습 Pearson 0.132 회복. **검증 세트 평가와 신체 결합 검증은 미완** → 7.3 의 "남은 검증" 1–6.
- 산출물은 `runs/phase1d/` (git 제외). 다른 컴퓨터로 옮길 때 `runs/` 전체를 복사할 것.

## 3c. 2026-09-03 (Ubuntu 이전): Linux 환경 구축
- 기기: Ubuntu, 16코어, 59 GB RAM, RTX 5090. `runs/`(635 MB, phase1d 포함) 복사본 사용 → Phase 0 재실행 없음.
- `uv sync` 가 수정 없이 성공(Python 3.12.14, JAX 0.4.38 CPU, torch 2.2.2+cu121). 테스트 19개 통과 31 s (Mac 2분).
- 문제 1개: sentence-transformers 가 CUDA 를 자동 선택하는데 torch 2.2.2 는 RTX 5090(sm_120) 커널이 없어 `no kernel image` 오류. → `translator.py` 에 `torch_device()` 추가(torch 빌드의 arch 목록에 GPU 가 있을 때만 cuda). 결과 CPU 사용.
- `phase0_c302_reference.py` 의 macOS 전용 경로(xcrun, /usr/local/opt/openjdk@21)를 `sys.platform` 으로 분기. Linux 는 시스템 g++·PATH 의 java. 이 기기에는 gcc·java 미설치(Phase 0 재실행 시 apt 필요).
- 검증: 웹 UI 웹소켓으로 "왼쪽으로 가" → turn_left(AVB+SMDV) 인식, 20 s 에 534 프레임(시뮬 26.7 s, 실시간 이상). 번역기 재학습은 Mac 과 수치 동일(minilm 0.837 ± 0.025) — `runs/phase4/train_linux_2026-09-03.txt`, 가중치는 Mac 원본 유지.

### GPU 전환 (같은 날 오후)
- 사용자 질문 "GPU로 하면 더 빠르다던데?" → 측정. `phase1d_fit.py` 3스텝: Mac 138 s/스텝, 이 CPU 43 s, **GPU 2 s** (vmap 32 자극 × 3000 스텝 역전파). 반면 float64 단일 궤적(테스트 19개): CPU 23 s, GPU 74 s. 즉 **배치 적합은 GPU, 대화형·테스트는 CPU**.
- pyproject: Linux 에서만 `jax[cuda12]==0.7.1`(0.7.2+ 는 numpy 2 요구), torch 는 CPU 휠(`[tool.uv.sources]` pytorch-cpu 인덱스; cu121 휠의 cuDNN 8.9 가 CUDA JAX 와 충돌). Mac 은 JAX 0.4.38 그대로. JAX 0.7.1 에서 코드 수정 없이 동작, 손실값 CPU 와 소수점 4자리 일치.
- `worm/server/app.py` 와 `tests/conftest.py` 는 `JAX_PLATFORMS=cpu` 기본. 실험 스크립트는 GPU. 동시 실행 시 `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

### Phase 1d 마무리: fit_F 검증 → 채택 안 함, 원인 분리 (RESULTS_PHASE1 7.3–7.4)
- 발견: fit_E/F 의 학습 분할은 fit_C 와 다르다(문서 오기 정정). 비교는 학습 집합 합집합을 뺀 공통 검증 세트에서.
- fit_F: 검증 Spearman 0.021 (Vfit 0.123, V0 0.086), 억제 재현 = 기저율, 회로 방향은 반대쪽 명령 뉴런 억제로 얻은 것. 신체 결합(`phase1d_body_check.py`): 전진·조향·RIS 유지, 후진은 게이트 3 mV 에 걸림(1.5 mV 면 회복), 터치는 10 pA 에도 무반응.
- fit_G1/G10 (λ_prior 1/10, GPU 각 5분): 검증 Spearman 0.002/0.009. θ가 거의 안 움직인 G10 도 붕괴 → **적합 전 초기값(억제 동작점 −50 mV)이 이미 0.018**. 원인은 억제를 켜는 것 자체.
- 결론: Vfit 유지. 억제 재현·감각→명령 선택은 시냅스 파라미터로 불가 — 부정 결과로 종결.

### Phase 3 개정: 유턴·춤·회전 강화 (ADR-016, RESULTS_PHASE3_4 1b)
- 진단: omega_turn 에서 RIV +55 / SMDV +51 mV 로 뉴런은 충분히 반응. 문제는 운동층 — 머리 편향(6 행)은 진행 방향을 못 바꾸고, 정적 배쪽 편향은 몸을 말지만 파동이 멈춤(+42° 최대).
- 채택: RIV ΔV > 30 mV 시 깊은 배쪽 굽힘 펄스(σ 4 행, 이득 0.36)가 머리→꼬리로 2.5 s 전파 (`Worm.k_omega`, Croll 1975/Broekmans 2016). omega_turn **+3° → +174°**, 전진 0.10 mm/s 유지. 문턱 30 mV 는 SMD 자극 시 갭정션으로 새는 RIV 상승(≈20 mV)을 걸러 다른 프로토콜 불변.
- 회전 시 1 Hz 문제: 편향 6 행 vs 수용기 6 행 이동 창의 불일치가 원인. 편향 12 행 + 이득 0.03 → 주파수 0.55–0.67 Hz, 회전 ±27° → +52°/−44°.
- local_search 재정의(기존 정의는 RIV 미사용): 후진 1 s → 오메가 2 s → 머리 흔들며 전진 3 s, 6 s 주기 → 12 s 에 +369°(피루엣 2회).
- 번역기: "유턴해/유턴/U턴" 이 forward 로 읽히던 문제 → 유턴 변형 7문장 추가(검증 0.834 → 0.858). 혼합 확률 0.45 가 RIV 진폭을 희석해 문턱을 못 넘는 문제 → 디코딩 선명화 p² (`predict(gamma=2)`, ADR-014 보완). 행동 평가: 프로토콜 일치 0.96 → 1.00, 축 일치 0.96 → 1.00, 최근접 0.65 → 0.81. 서버에서 "유턴해" → 150° 회전 확인.
- 산출물: `experiments/phase3_omega.py`, `runs/phase3/omega_scan*.txt`, `turn_freq_scan.txt`, `turn_rows12_scan.txt`, `protocols_omega_check.txt`, `omega_amp_scan.txt`, `protocols_final.txt`(개정 전 `*_before_omega.*`), `runs/phase4/train.txt`, `behavior_eval.txt`. 테스트 19개 통과.

### 첫 프로젝트 마무리
- README 를 영어로 다시 썼고(`README.md`), 한국어판은 `README.ko.md` 로 분리해 상호 링크. 나머지 문서는 한국어 그대로.

## 4. 미해결 과제 (우선순위)
1. ~~억제 재현과 감각→명령 선택(클래스별 적합)~~ → 2026-09-03 종결(부정 결과, RESULTS_PHASE1 7.3–7.4). 감각→명령 방향 선택이 필요하면 시냅스 파라미터가 아닌 층(감각 뉴런 극성 가정, 운동층 게이트 규칙)에서 다시 설계.
2. ~~유턴·춤의 회전 강화, 회전 시 파동 주파수 상승(1 Hz) 조정~~ → 2026-09-03 완료 (ADR-016, RESULTS_PHASE3_4 1b). 남은 것: 오메가 펄스의 전파 시간·폭은 탐색값(문헌 정량값 없음); head_sweep 의 1 Hz 는 프로토콜 정의(0.5 s 교대)에 따른 것.
3. 실행 로그 저장/재생, B-2(시뮬레이터 루프 미세조정, 무경사).
4. 논문 초고 (`docs/PAPER_OUTLINE.md`).

## 5. 다른 컴퓨터에서 이어가기
- macOS Intel: README 의 우회(JAX 0.4.38, nrnivmodl CXX, openjdk@21) 필요. Apple Silicon/Linux: JAX·torch 최신으로 올려도 됨(pyproject 의 고정 해제).
- 2026-09-03 두 번째 Intel Mac 으로 이전: `runs/` 복사본으로 Phase 0 재생성 없이 테스트 18개·웹 UI 정상. uv 0.4.24 → 0.12.9 업그레이드, `requires-python <3.13` 추가(uv.lock 갱신). brew openjdk@21 은 이 기기에서 권한 오류로 미설치 — Phase 0 재실행 시에만 필요.
- Linux(Ubuntu): `uv sync` 만으로 된다(2026-09-03 확인, 3c 절). JAX 는 CUDA 빌드(0.7.1)가 깔리고 적합 스크립트는 GPU 를 쓴다. 서버·테스트는 CPU 기본. torch 는 CPU 휠(임베딩 1분). Phase 0 재실행이 필요하면 `apt install build-essential openjdk-21-jre-headless` (README Linux 절). `runs/`(약 650 MB, phase1d 포함)를 함께 복사하면 재생성 불필요.
- Claude Code 로 이어갈 때: 이 폴더의 `CLAUDE.md` 가 자동으로 읽힌다. 대화 원본을 다시 보려면 `history/*.jsonl`.
