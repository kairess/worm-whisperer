# 논문 개요 (초안, 2026-09-02)

## 가제
A differentiable reimplementation of the OpenWorm c302 nervous-system model: bistability, gap-junction-dominated signal propagation, and comparison with the *C. elegans* signal propagation atlas

## 한 줄 주장
커넥톰을 고정한 채 OpenWorm c302 C2 동역학을 미분 가능하게 재구현하고 실측 신호전파 아틀라스(Randi 2023)와 비교하면, (1) 동역학 모델이 해부학적 연결 강도보다 기능을 유의하게 잘 예측하지만, (2) 그 예측력은 갭정션 전기 확산에서 나오고 화학 시냅스는 휴지전위에서 사실상 꺼져 있으며, (3) 상관 지표는 이미 측정 재현성 상한에 근접해 있고, (4) 억제 응답과 이동 행동은 현재 세포 모델로는 재현되지 않는다.

## 결과 절 (각 절의 근거 문서)
1. **재구현과 검증** — NEURON 대비 뉴런 전압 RMS 0.001 mV, 두 커넥톰(White/Varshney 계열, Cook 2019), dt 수렴, 8–12배 속도. (`docs/RESULTS_PHASE1.md` 1절)
2. **전역 쌍안정성** — 지속 감각 입력 ≥ 2 pA 에서 전 뉴런이 흥분성 역전전위로 고정. 문턱과 진입 시간의 자극 의존성. 부분회로 vs 전체 네트워크. (2–3절)
3. **아틀라스 비교** — 172 자극 뉴런, 23,883 쌍. AUROC 0.63 vs 해부학 0.545 (부트스트랩 CI). Spearman 0.115 vs 측정 상한 0.14. (5절)
4. **경로 절제** — 갭정션만: AUROC 0.613, 화학만: 0.552. 화학 시냅스 활성도 s ≈ 10⁻⁵ 의 정량적 이유. 억제 재현율 0. (5절 결과 2–3)
5. **극성·동작점 변형과 파라미터 적합** — CeNGEN 극성(억제 50%), Vth 이동, 전도도 스케일: 안정성은 크게 바뀌지만 예측력은 불변. 미분 가능 시뮬레이터로 전역 파라미터 6개를 적합하면(학습 32/검증 140 자극 뉴런) 검증 Spearman 0.131 (상한 0.14), 폭주 소멸, 터치 회로 논리 재현. 최적화는 억제를 제거한다. (5–6절)
6. **행동으로의 확장** — c302 세포 모델로는 이동 불가(고유수용 두 방식, 고립 세포 진동 부재, 채널 확장 탐색). Boyle 2012 운동층으로 교체하면 커넥톰의 명령 개재뉴런 출력(AVB/AVA)이 전진/후진 기어가기를 만든다(0.18 / −0.14 mm/s, 0.5 / 0.4 Hz). SMD 조향으로 ±26° 회전, RIS 로 정지. 감각 자극(터치, ASH, 화학주성)은 명령 선택에 실패. (`docs/RESULTS_PHASE2.md`, `RESULTS_PHASE3_4.md`)
7. **자연어 → 뉴런 자극 번역기** — 화이트리스트 19채널, 프로토콜 혼합 헤드, 다국어 임베딩 고정. 새 표현 정확도 0.84, 행동 수준 평가. LLM 은 근육/운동뉴런에 접근할 수 없다는 구조적 제약. (`RESULTS_PHASE3_4.md` 3절)

## 그림 후보
- Fig 1: 파이프라인 (NeuroML → JAX), 검증 오버레이 (`runs/phase0/*/…jax_vs_neuron.png`)
- Fig 2: 쌍안정성 — 자극 세기별 네트워크 평균 전압 시간 경과 (`runs/phase1/runaway_*.png`)
- Fig 3: 아틀라스 비교 — 산점도, AUROC/Spearman 막대, ROC (`runs/phase1b/fig_sigprop.png`), 상한선 추가 필요
- Fig 4: 경로 절제와 억제 재현율
- Fig 5: 신체 결합 — c302 운동층(이동 없음) vs Boyle 운동층(전진/후진 파동) 키모그래프 (`runs/phase2/fig_kymograph.png`, `fig_integrated.png`)
- Fig 6: 명령 → 행동 표 (프로토콜 13개 기술자), 번역기 혼합 확률 예시, UI 스크린샷

## 두 번째 논문 후보 (응용)
"Language-grounded control of a connectome-constrained *C. elegans* model": 번역기 + 화이트리스트 + Boyle 운동층 시스템. 핵심 주장은 LLM 표상이 실험 프로토콜 공간에 접지되며 커넥톰이 인과 경로로 남는다는 것. 한계: 감각 경로 미재현, 운동층은 문헌 모델 대체.

## 한계와 정직한 서술
- 광자극 ↔ 정전류 2 pA 대응은 가정. 응답 창 30 s 대 2 s.
- 아틀라스는 머리 뉴런 위주(운동뉴런 미측정). 절제 실험 검증은 불가.
- CeNGEN 극성 예측의 상충(1,085개)과 미예측(447개) 처리 정책이 결과에 미치는 영향은 "split" vs "default" 두 가지만 검토.
- 신경펩타이드/모노아민 층(Bentley 2016)은 미포함. Randi 등은 펩타이드 신호가 기능 연결의 상당 부분을 설명한다고 보고.

## 저널/학회 후보
PLoS Computational Biology, eNeuro, Journal of Computational Neuroscience; 워크숍: OpenWorm 커뮤니티, COSYNE 포스터.
