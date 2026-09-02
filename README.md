# worm-whisperer

**Grounding Language in a Connectome** — 자연어 명령을 *C. elegans* 커넥톰을 통해 행동으로 잇는 시뮬레이터.
OpenWorm의 302뉴런 모델(c302)의 연결 구조를 **그대로 유지**하고, 언어 모델은 근육이나 운동뉴런을 직접 건드리지 않는다.
번역기가 하는 일은 **"가상 실험자"** 역할이다. 실제 논문에 보고된 광유전학 / 감각 자극 프로토콜 중 하나를 골라
특정 뉴런에 전류를 주고, 나머지는 커넥톰과 신체 물리가 만든다.

```
"앞으로 가"     → AVB 명령 개재뉴런 자극 → 커넥톰 → 운동층 → 전진 0.18 mm/s, 0.5 Hz
"왼쪽으로 가"   → AVB + SMDV 머리 조향 뉴런 → 12초에 +27° 좌회전
"유튜브 보는중" → RIS 수면활성뉴런 자극 → 운동 정지 (정지 비율 0.78)
```

작동하지 않는 것도 기록한다. 감각 자극(터치, 기피, 화학주성)은 방향을 고르지 못하고, 운동층은 c302 세포 모델로
파동이 창발하지 않아 문헌 모델(Boyle 2012)로 대체했다. 자세한 내용은 결과 문서와 [docs/SESSION_LOG.md](docs/SESSION_LOG.md).

| 문서 | 내용 |
|------|------|
| [PLAN.md](PLAN.md) | 목표, 아키텍처, 단계별 로드맵, 리스크 |
| [docs/COMMANDS.md](docs/COMMANDS.md) | 명령어 ↔ 뉴런 자극 매핑표와 근거 논문 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 참고 문헌, 도구, 데이터셋 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 설계 결정 기록 (ADR) |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Phase 0 기준 실행과 C2 모델 사양 |
| [docs/RESULTS_PHASE1.md](docs/RESULTS_PHASE1.md) | Phase 1 결과: JAX 재구현 검증, 쌍안정성, 신호전파 아틀라스 비교, 파라미터 적합 |
| [docs/BODY_MODEL.md](docs/BODY_MODEL.md) | 2D 신체 모델 사양과 검증 |
| [docs/RESULTS_PHASE2.md](docs/RESULTS_PHASE2.md) | Phase 2 결과: 신경–신체 결합, 고유수용 피드백, 진동자 탐색 |
| [docs/RESULTS_PHASE3_4.md](docs/RESULTS_PHASE3_4.md) | Phase 3–5 결과: 프로토콜, 조향/정지, 번역기, UI, 화학주성 |
| [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | 논문 개요 초안 |
| [docs/SESSION_LOG.md](docs/SESSION_LOG.md) | 작업 기록(대화 요약)과 이어가기 안내. 원본 대화는 `history/*.jsonl` |
| [CLAUDE.md](CLAUDE.md) | Claude Code 가 자동으로 읽는 프로젝트 지침 |

## 환경 구축 (macOS Intel 기준)

```bash
brew install openjdk@21                       # jNeuroML(NeuroML → NEURON 변환)에 필요
uv sync                                       # Python 3.12, c302, pyNeuroML, NEURON, JAX 0.4.38
uv run python experiments/phase0_c302_reference.py --full   # 전체 302뉴런 기준 실행, runs/phase0/ 에 출력
```

데모 실행:
```bash
uv run python experiments/phase4_train_translator.py     # 번역기 학습 (runs/phase4/translator.pt, 1회)
uv run uvicorn worm.server.app:app --port 8000            # http://localhost:8000 에서 "앞으로 가", "왼쪽으로 가", "춤춰봐", "유튜브 보는중"
uv run pytest -q tests                                    # 테스트 (통합 테스트 포함 약 2분)
```

알려진 문제와 우회 (스크립트에 이미 반영됨):
- JAX는 0.4.38이 Intel Mac을 지원하는 마지막 버전이라 고정. Apple Silicon이면 최신으로 올려도 됨.
- `pynml -neuron -run`은 pip 설치 NEURON을 못 찾는다. 스크립트는 `-neuron`으로 내보내기만 하고, `nrnivmodl`로 직접 컴파일한 뒤 생성된 `LEMS_*_nrn.py`를 실행한다.
- `nrnivmodl` 컴파일 시 `'cstddef' file not found`가 나면 CommandLineTools의 깨진 libc++ 헤더 때문. 스크립트는 `CXX="clang++ -I$(xcrun --show-sdk-path)/usr/include/c++/v1"`로 우회한다.
- Java는 `/usr/local/opt/openjdk@21/bin`을 PATH 앞에 넣는다 (스크립트 내부에서 처리).
