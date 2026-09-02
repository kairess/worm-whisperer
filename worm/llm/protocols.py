"""프로토콜 화이트리스트 (docs/COMMANDS.md 의 코드화). LLM/번역기는 이 목록 밖의 뉴런을 자극할 수 없다.

각 프로토콜은 (이름, 등급, 자극 스케줄 생성 함수, 근거) 로 정의된다. 스케줄: [(t0_s, t1_s, {뉴런: pA}), ...]
채널 화이트리스트 CHANNELS: 감각뉴런 쌍, 광유전학으로 검증된 명령/조향/수면 뉴런. 운동뉴런·근육 없음.
"""
CHANNELS = {
    # 감각 (S)
    "PLM": ["PLML", "PLMR"], "ALM": ["ALML", "ALMR"], "AVM": ["AVM"], "ASH": ["ASHL", "ASHR"],
    "ASE": ["ASEL", "ASER"], "AWC": ["AWCL", "AWCR"], "AWA": ["AWAL", "AWAR"], "ASK": ["ASKL", "ASKR"],
    # 명령 개재뉴런 (O)
    "AVB": ["AVBL", "AVBR"], "AVA": ["AVAL", "AVAR"], "PVC": ["PVCL", "PVCR"], "AVD": ["AVDL", "AVDR"],
    # 머리 조향 / 회전 (O)
    "SMDD": ["SMDDL", "SMDDR"], "SMDV": ["SMDVL", "SMDVR"], "RMDD": ["RMDDL", "RMDDR"], "RMDV": ["RMDVL", "RMDVR"], "RIV": ["RIVL", "RIVR"],
    # 수면/상태 (O+X)
    "RIS": ["RIS"], "ALA": ["ALA"],
}
ALLOWED_NEURONS = sorted({n for v in CHANNELS.values() for n in v})

def _stim(chs, amp):
    return {n: amp for c in chs for n in CHANNELS[c]}

def forward(amp=5.0, dur=10.0):        return [(0, dur, _stim(["AVB"], amp))]
def reverse(amp=5.0, dur=10.0):        return [(0, dur, _stim(["AVA"], amp))]
def stop(dur=10.0):                    return []
def turn_left(amp=5.0, dur=10.0):      return [(0, dur, {**_stim(["AVB"], amp), **_stim(["SMDV"], amp)})]     # 규약: 배쪽 굽힘 = 반시계 = 왼쪽 (화면 y 위쪽 기준; 벌레가 왼쪽 옆면으로 누운 자세)
def turn_right(amp=5.0, dur=10.0):     return [(0, dur, {**_stim(["AVB"], amp), **_stim(["SMDD"], amp)})]
def omega_turn(amp=5.0, dur=10.0):     return [(0, 2, _stim(["AVA"], amp)), (2, 4, {**_stim(["AVB"], amp), **_stim(["SMDV", "RIV"], amp)}), (4, dur, _stim(["AVB"], amp))]
def local_search(amp=5.0, dur=12.0):   # 후진 1 s + 오메가 1 s + 전진 2 s 반복 (Gray 2005; Hills 2004)
    sch = []; t = 0.0; side = ["SMDD", "SMDV"]; i = 0
    while t < dur:
        sch += [(t, t + 1, _stim(["AVA"], amp)), (t + 1, t + 2, {**_stim(["AVB"], amp), **_stim([side[i % 2]], amp)}), (t + 2, t + 4, _stim(["AVB"], amp))]; t += 4; i += 1
    return sch
def head_sweep(amp=5.0, dur=10.0):     # 1 Hz 머리 좌우 흔들기 (foraging)
    sch = [(0, dur, _stim(["AVB"], amp))]; t = 0.0; i = 0
    while t < dur: sch.append((t, t + 0.5, _stim([["SMDD", "SMDV"][i % 2]], amp))); t += 0.5; i += 1
    return sch
def quiescence_RIS(amp=5.0, dur=10.0): return [(0, dur, _stim(["AVB"], amp)), (3, dur, _stim(["RIS"], 2 * amp))]   # 전진 중 RIS 활성 → 정지
def quiescence_ALA(amp=5.0, dur=10.0): return [(0, dur, _stim(["AVB"], amp)), (3, dur, _stim(["ALA"], 2 * amp))]
def escape(amp=5.0, dur=10.0):         return [(0, 1.0, _stim(["ASH"], amp)), (1.0, dur, {})]
def forward_touch(amp=5.0, dur=10.0):  return [(0, 1.0, _stim(["PLM"], amp))]
def reverse_touch(amp=5.0, dur=10.0):  return [(0, 1.0, _stim(["ALM", "AVM"], amp))]

PROTOCOLS = {
    "forward": ("O", forward, "Chalfie 1985: AVB 전진 명령"),
    "reverse": ("O", reverse, "Chalfie 1985; Guo 2009: AVA 광활성화 → 후진"),
    "stop": ("-", stop, "기준 상태"),
    "turn_left": ("O", turn_left, "Gray 2005: SMD 머리 조향 (SMDV, 배쪽 = 왼쪽 규약)"),
    "turn_right": ("O", turn_right, "Gray 2005: SMD 머리 조향"),
    "omega_turn": ("O", omega_turn, "Gray 2005: RIV/SMDV 배쪽 오메가턴"),
    "local_search": ("O", local_search, "Gray 2005; Hills 2004: 후진+오메가턴 반복"),
    "head_sweep": ("O", head_sweep, "탐색 시 머리 흔들기"),
    "quiescence_RIS": ("O+X", quiescence_RIS, "Turek 2016: RIS 광활성화 → 운동 정지 (펩타이드 확장 플래그)"),
    "quiescence_ALA": ("O+X", quiescence_ALA, "Hill 2014: ALA 스트레스 수면"),
    "escape": ("S", escape, "Pirri 2009: ASH → 후진 → 오메가턴"),
    "forward_touch": ("S", forward_touch, "Chalfie 1985: 후방 터치 → 전진"),
    "reverse_touch": ("S", reverse_touch, "Chalfie 1985: 전방 터치 → 후진"),
}

def validate(schedule):
    for t0, t1, stim in schedule:
        bad = [n for n in stim if n not in ALLOWED_NEURONS]
        if bad: raise ValueError(f"whitelist violation: {bad}")
    return schedule
