# 명령 ↔ 프로토콜 ↔ 뉴런 매핑

LLM이 선택할 수 있는 프로토콜의 전체 목록이다. 이 표에 없는 뉴런은 LLM이 자극할 수 없다.
"등급"은 과학적 근거의 성격을 뜻한다.
- **O** (optogenetic): 실제 광유전학 실험으로 해당 뉴런 활성화 → 행동이 보고됨. 시뮬레이션에서는 전류 주입.
- **S** (sensory): 감각뉴런에 자연 자극(화학, 접촉)을 준다. 행동은 회로에서 창발해야 함. 더 어렵고 더 의미 있음.
- **X** (extension): 커넥톰 밖의 요소(신경펩타이드) 추가가 필요. 플래그로 분리.

## 1. 이동

| 명령 예시 | 프로토콜 | 자극 대상 | 등급 | 파라미터 | 근거 |
|-----------|----------|-----------|------|----------|------|
| 앞으로 가, 전진 | `forward` | AVBL/AVBR (+PVC) 지속 전류 | O | amp, dur | Chalfie 1985: AVB/PVC 전진 명령 개재뉴런 |
| 앞으로 가 (감각) | `forward_touch` | PLML/PLMR 펄스 (뒤쪽 접촉) | S | dur | Chalfie 1985: 후방 터치 → 전진 가속 |
| 뒤로 가, 후진 | `reverse` | AVAL/AVAR (+AVD/AVE) 지속 전류 | O | amp, dur | Chalfie 1985; Guo 2009 (AVA ChR2 → 후진) |
| 뒤로 가 (감각) | `reverse_touch` | ALML/ALMR/AVM 펄스 (앞쪽 접촉) | S | dur | Chalfie 1985: 전방 터치 → 후진 |
| 멈춰 | `stop` | 모든 자극 해제 | - | - | 기준 상태 |

## 2. 회전 / 방향

*C. elegans*는 옆으로 누워 기므로 등쪽(dorsal)/배쪽(ventral) 굽힘이 평면에서 좌/우 회전이 된다. 어느 쪽이 "왼쪽"인지는 벌레가 어느 옆면으로 누웠는지에 따라 정해지며, 시뮬레이터에서 초기 자세로 고정한다.

| 명령 예시 | 프로토콜 | 자극 대상 | 등급 | 파라미터 | 근거 |
|-----------|----------|-----------|------|----------|------|
| 왼쪽/오른쪽으로 가 (환경) | `chemotaxis_toward` | 가상 NaCl 소스를 목표 방향에 배치 → ASEL/ASER 농도 입력 | S | 방향, 농도 | Pierce-Shimomura 1999 (피루엣), Iino & Yoshida 2009 (klinotaxis) |
| 왼쪽/오른쪽으로 가 (광유전학) | `head_steer` | SMDD 또는 SMDV (+RMDD/RMDV) 편측 지속 전류 | O | 측, amp | Gray, Hill, Bargmann 2005 (SMD/RIV 회전 회로); Hendricks 2012 (RIA 머리 굽힘) |
| 확 돌아, 유턴 | `omega_turn` | AVA 짧은 펄스 → RIV/SMDV 펄스 | O | - | Gray 2005: RIV 배쪽 오메가턴; Pirri 2009 (탈출 반응) |
| 도망가 | `escape` | ASHL/ASHR 강한 펄스 (기피 자극) | S | amp | Pirri 2009: ASH → AVA 후진 → 오메가턴 순서가 창발해야 함 |

## 3. 탐색 / 춤

| 명령 예시 | 프로토콜 | 자극 대상 | 등급 | 파라미터 | 근거 |
|-----------|----------|-----------|------|----------|------|
| 춤춰봐, 꿈틀거려 | `local_search` | AVA 1 s → AVB+RIV+SMDV 2 s(오메가턴) → AVB+SMDD/SMDV 각 1.5 s(머리 흔들며 전진), 6 s 주기 반복 (피루엣) | O | 반복 횟수, 간격 | Gray 2005; Hills 2004 (먹이 제거 후 국소 탐색: 후진/오메가턴 빈도 급증) |
| 춤춰봐 (감각) | `food_removed` | AWC/ASK "off 반응" 펄스 (먹이 사라짐 신호) | S | - | Gray 2005: AWC/ASK 글루타메이트 → 국소 탐색 창발 |
| 두리번거려 | `head_sweep` | SMDD/SMDV 교대 저주파 자극 (1 Hz) | O | 주파수 | 먹이 탐색 시 머리 흔들기 (foraging head swings) |
| 헤엄쳐 | `swim` | 환경 항력비를 1.5로 (액체) + `forward` | S | - | Pierce-Shimomura 2008: 수영은 기어가기와 다른 보행(gait), 주파수 ~2 Hz |

## 4. 정지 / 휴식 상태

| 명령 예시 | 프로토콜 | 자극 대상 | 등급 | 파라미터 | 근거 |
|-----------|----------|-----------|------|----------|------|
| 유튜브 보는중, 쉬어, 자 | `quiescence_RIS` | RIS 지속 전류 + FLP-11 펩타이드 억제 층 ON | O+X | amp, dur | Turek 2013, 2016: RIS는 수면활성 뉴런, 광활성화 시 즉시 운동 정지; FLP-11 매개 |
| 피곤해 (스트레스 수면) | `quiescence_ALA` | ALA 지속 전류 + FLP-13 층 ON | O+X | amp, dur | Hill 2014; Nelson 2014: ALA 스트레스 유발 수면 |
| 밥 먹는중, 어슬렁 | `dwelling` | NSM 세로토닌 층 ON (속도 저하, 잦은 짧은 후진) | X | - | Flavell 2013: 세로토닌 → dwelling, PDF → roaming |
| 돌아다녀 | `roaming` | PDF 층 ON + `forward` | X | - | Flavell 2013 |

"꼼지락거림"은 별도 자극이 아니다. RIS 자극으로 운동뉴런 출력이 억제된 상태에서 남는 잡음성 근육 활성이 자연스럽게 작은 움직임을 만든다. 만약 완전히 멈춰버리면 그것도 기록한다 (실제 수면 중에도 미세 움직임이 있음, Iwanir 2013).

## 5. 실험 (사용자 확인 필요)

| 명령 예시 | 프로토콜 | 동작 | 근거 |
|-----------|----------|------|------|
| AVA 없이 뒤로 가봐 | `ablate` + `reverse` | 지정 뉴런의 시냅스 출력 0으로 | Chalfie 1985 레이저 절제 |
| 원래대로 | `restore` | 절제 해제 | - |
| 커넥톰 바꿔 (Witvliet L1) | `swap_connectome` | 재생성 필요, 세션 재시작 | Witvliet 2021 |

## 6. LLM 응답 형식

```json
{
  "protocol": "quiescence_RIS",
  "params": {"amp_nA": 0.5, "dur_ms": 30000},
  "status_label": "유튜브 보는중",
  "rationale": "정지 상태 요청. RIS 광활성화가 즉각적 운동 정지를 유도함 (Turek 2016). 펩타이드 확장 층 사용.",
  "extension_used": true
}
```

`extension_used`가 true이면 UI에 "커넥톰 외 확장 사용" 배지가 뜬다.
