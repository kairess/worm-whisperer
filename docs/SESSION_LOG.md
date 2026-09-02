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
9. **Phase 4** 번역기(MiniLM 임베딩 → 프로토콜 혼합, ADR-014): 새 표현 0.84, 행동 축 일치 0.96.
10. **Phase 5** FastAPI+WebSocket 웹 UI.
11. fit_D(억제·터치 방향 제약 추가) 실패 → 전역 파라미터로는 불가.
12. 사용자 피드백: 좌/우 반대 → 규약 수정(배쪽 SMDV = 왼쪽). 코 터치 시 전진은 모델 한계(억제 부재)로 설명.

## 3. 현재 상태
- 실행: `uv sync` 후 `uv run python experiments/phase0_c302_reference.py --full` (기준 네트워크 생성, NEURON/Java 필요) → `uv run python experiments/phase4_train_translator.py` → `uv run uvicorn worm.server.app:app --port 8000`.
- 테스트: `uv run pytest -q tests` (18개, 약 2분).
- `runs/` 는 git 제외(약 700 MB). 다른 컴퓨터에서는 Phase 0 스크립트로 `runs/phase0/c302_C2_LW_Full_avb-ava/` 를 재생성해야 모든 실험이 돈다. 번역기 가중치(`runs/phase4/translator.pt`)는 재학습 1분.

## 4. 미해결 과제 (우선순위)
1. 억제 재현과 감각→명령 선택: 시냅스 클래스별 파라미터 적합 (Phase 1c 확장).
2. 유턴·춤의 회전 강화, 회전 시 파동 주파수 상승(1 Hz) 조정.
3. 실행 로그 저장/재생, B-2(시뮬레이터 루프 미세조정, 무경사).
4. 논문 초고 (`docs/PAPER_OUTLINE.md`).

## 5. 다른 컴퓨터에서 이어가기
- macOS Intel: README 의 우회(JAX 0.4.38, nrnivmodl CXX, openjdk@21) 필요. Apple Silicon/Linux: JAX·torch 최신으로 올려도 됨(pyproject 의 고정 해제).
- Claude Code 로 이어갈 때: 이 폴더의 `CLAUDE.md` 가 자동으로 읽힌다. 대화 원본을 다시 보려면 `history/*.jsonl`.
