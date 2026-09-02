# worm-whisperer (Grounding Language in a Connectome) — Claude Code 작업 지침

이 프로젝트는 OpenWorm c302 커넥톰 모델을 JAX로 재구현하고, 자연어 명령을 뉴런 자극 프로토콜로 번역해 2D 벌레를 움직이는 연구 프로젝트다.

## 먼저 읽을 것
1. `docs/SESSION_LOG.md` — 지금까지의 작업 요약과 미해결 과제
2. `PLAN.md` — 목표, 원칙(1.1절), 단계 상태
3. `docs/DECISIONS.md` — ADR 1–14 (왜 그렇게 했는지)
4. 결과: `docs/RESULTS_PHASE1.md`, `RESULTS_PHASE2.md`, `RESULTS_PHASE3_4.md`

## 지켜야 할 원칙
- 커넥톰(연결의 존재/개수)은 절대 바꾸지 않는다. 바꾸는 것은 동역학 가정이며 변형 번호(V0, V1, Vfit, C1…)로 관리한다.
- LLM/번역기는 `worm/llm/protocols.py` 의 화이트리스트 밖 뉴런(운동뉴런, 근육)을 자극할 수 없다.
- 새 기전은 근거 논문과 함께 docs 에 기록한다. 부정 결과도 기록한다.
- 보고 수치는 dt 0.05 ms float64 로 재확인 (탐색은 dt 0.25).
- 사용자는 한국어로 소통하며, 자율 진행과 솔직한 한계 보고를 원한다.

## 실행
```bash
uv sync
uv run python experiments/phase0_c302_reference.py --full     # runs/phase0/ 기준 네트워크 (Java, NEURON 필요; README 우회 참고)
uv run python experiments/phase4_train_translator.py           # 번역기 가중치
uv run uvicorn worm.server.app:app --port 8000                 # 데모 UI
uv run pytest -q tests                                         # 테스트 18개
```

## 구조
- `worm/neural/` connectome.py(NeuroML 파서), jaxsim.py(C2 시뮬레이터), variants.py(V0/V1/V2/Vfit, θ), readers/
- `worm/body/` rod2d.py(2D 막대), boyle_motor.py(Boyle 2012 운동층), muscle_map.py
- `worm/sim.py` 신경–신체 결합(Worm), 행동 기술자
- `worm/llm/` protocols.py(화이트리스트), phrases.py, translator.py
- `worm/server/app.py`, `web/index.html` UI
- `experiments/` 단계별 재현 스크립트, `runs/`(git 제외) 결과
