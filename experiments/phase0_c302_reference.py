"""Phase 0: c302 모델을 생성하고 NEURON으로 실행해 기준 전압 파일을 만든다.

사용 예:
  uv run python experiments/phase0_c302_reference.py                     # 운동 부분회로, AVB→AVA 자극
  uv run python experiments/phase0_c302_reference.py --stim none         # 무자극 대조군
  uv run python experiments/phase0_c302_reference.py --cells AVAL --stim step   # 고립 단일 뉴런 계단 전류
  uv run python experiments/phase0_c302_reference.py --full              # 302 뉴런 전체
출력: runs/phase0/<reference>/  (NeuroML, LEMS, NEURON 코드, *.dat)
"""
import argparse, importlib, os, subprocess, sys, time, shutil
import c302
sys.path.insert(0, os.getcwd())
import neuroml.writers as writers

if sys.platform == "darwin":                              # Intel Mac 우회 (README 참고)
    SDK = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    CXX = f"/usr/bin/clang++ -I{SDK}/usr/include/c++/v1"   # 툴체인의 깨진 libc++ 헤더 우회
    JAVA_BIN = "/usr/local/opt/openjdk@21/bin"
else:                                                     # Linux: 시스템 g++ 와 PATH 의 java 사용 (apt: build-essential openjdk-21-jre-headless)
    CXX = None
    JAVA_BIN = None

def range_incl(a, b):
    return range(a, b + 1)

MOTORS = ([f"DA{i}" for i in range_incl(1, 9)] + [f"DB{i}" for i in range_incl(1, 7)]
          + [f"VA{i}" for i in range_incl(1, 12)] + [f"VB{i}" for i in range_incl(1, 11)]
          + [f"DD{i}" for i in range_incl(1, 6)] + [f"VD{i}" for i in range_incl(1, 13)]
          + [f"AS{i}" for i in range_incl(1, 11)])
COMMAND = ["AVAL", "AVAR", "AVBL", "AVBR", "PVCL", "PVCR", "AVDL", "AVDR", "AVEL", "AVER"]

def build(a, out_dir):
    ParameterisedModel = getattr(importlib.import_module(f"c302.parameters_{a.params}"), "ParameterisedModel")
    params = ParameterisedModel()
    if a.full:
        cells, tag = None, "Full"
    elif a.cells:
        cells, tag = a.cells.split(","), "Cells_" + "_".join(a.cells.split(",")[:3])
    else:
        cells, tag = COMMAND + MOTORS, "Motor"
    reader_tag = "" if a.reader == c302.DEFAULT_DATA_READER else "_" + a.reader.split(".")[-1].replace("cect_", "").replace("herm", "")
    reference = f"c302_{a.params}_LW_{tag}_{a.stim}{reader_tag}"
    muscles = (not a.no_muscles) and not a.cells
    nml_doc = c302.generate(
        reference, params, cells=cells, cells_to_plot=cells[:9] if cells else ["AVAL", "AVBL", "DB1", "VB1"],
        cells_to_stimulate=[], muscles_to_include=muscles, duration=a.duration, dt=a.dt,
        target_directory=out_dir, data_reader=a.reader, verbose=False,
    )
    d = a.duration
    if a.stim == "avb-ava":            # 0-40% AVB(전진), 50-100% AVA(후진)
        for c in ["AVBL", "AVBR"]:
            c302.add_new_input(nml_doc, c, "0ms", f"{int(d*0.4)}ms", a.amp, params)
        for c in ["AVAL", "AVAR"]:
            c302.add_new_input(nml_doc, c, f"{int(d*0.5)}ms", f"{int(d*0.5)}ms", a.amp, params)
    elif a.stim == "step":             # 첫 셀에 계단 전류 20-70%
        c302.add_new_input(nml_doc, (cells or ["AVAL"])[0], f"{int(d*0.2)}ms", f"{int(d*0.5)}ms", a.amp, params)
    elif a.stim != "none":
        raise SystemExit("unknown --stim")
    writers.NeuroMLWriter.write(nml_doc, os.path.join(out_dir, reference + ".net.nml"))
    return reference

def run_neuron(out_dir, reference):
    env = dict(os.environ, PATH=JAVA_BIN + ":" + os.environ["PATH"]) if JAVA_BIN else dict(os.environ)
    r = subprocess.run(["pynml", f"LEMS_{reference}.xml", "-neuron", "-nogui"], cwd=out_dir, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(out_dir, f"LEMS_{reference}_nrn.py")):
        print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit("pynml export failed")
    shutil.rmtree(os.path.join(out_dir, "x86_64"), ignore_errors=True)
    r = subprocess.run(["nrnivmodl"], cwd=out_dir, env=dict(env, CXX=CXX) if CXX else env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit("nrnivmodl failed")
    t0 = time.time()
    r = subprocess.run([sys.executable, f"LEMS_{reference}_nrn.py"], cwd=out_dir, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit("NEURON run failed")
    sim_s = [l for l in r.stdout.splitlines() if "Finished NEURON simulation" in l]
    print(f"[run] {reference}: wall {dt:.1f}s ({sim_s[0].strip() if sim_s else ''})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="C2")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--cells", default=None, help="쉼표 구분 셀 목록 (근육 제외)")
    ap.add_argument("--no-muscles", action="store_true")
    ap.add_argument("--stim", default="avb-ava", choices=["avb-ava", "step", "none"])
    ap.add_argument("--amp", default="10pA")
    ap.add_argument("--duration", type=float, default=1000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--reader", default=c302.DEFAULT_DATA_READER)
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    tmp = os.path.abspath("runs/phase0/_gen_" + str(os.getpid())); os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    ref = build(a, tmp)
    out_dir = os.path.abspath(f"runs/phase0/{ref}"); shutil.rmtree(out_dir, ignore_errors=True); os.rename(tmp, out_dir)
    print(f"[gen] {ref} in {time.time()-t0:.1f}s -> {out_dir}")
    if not a.no_run:
        run_neuron(out_dir, ref)
        print("[plot]", subprocess.run([sys.executable, "experiments/phase0_plot.py", out_dir, ref], capture_output=True, text=True).stdout[-1500:])

if __name__ == "__main__":
    main()
