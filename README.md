# worm-whisperer

**Grounding Language in a Connectome** — a simulator that turns natural-language commands into *C. elegans* behavior *through* the worm's connectome, without letting the language model touch the muscles.

*한국어 README: [README.ko.md](README.ko.md)* · **Live demo (recorded simulations, mazes and neuron readouts): [kairess.github.io/worm-whisperer](https://kairess.github.io/worm-whisperer/)**

## The idea

Most "LLM controls a robot" demos let the model drive the actuators directly. This project does the opposite. The wiring diagram of the OpenWorm c302 model (302 neurons, chemical synapses and gap junctions from the published connectome) is **never modified**, and the language model cannot stimulate motor neurons or muscles at all. Its only job is to act as a **virtual experimenter**: it picks one of the optogenetic / sensory stimulation protocols reported in the literature, injects current into a short whitelist of sensory, command and steering interneurons, and everything downstream — command selection, the motor pattern, the 2D body — is produced by the connectome dynamics and the body physics.

```
"go forward"       → AVB command interneurons        → connectome → motor layer → crawling at 0.18 mm/s, 0.56 Hz
"turn left"        → AVB + SMDV head-steering neurons → +52° heading change in 12 s
"make a U-turn"    → AVA (reverse) → RIV/SMDV omega turn → AVB (forward) → +174° reorientation
"dance"            → repeated reversal + omega turn + head sweeps (pirouettes, +369° in 12 s)
"watching YouTube" → RIS sleep-active neuron          → locomotion stops (quiescent 78 % of the time)
```

Everything that *fails* is recorded too. Sensory protocols (touch, aversive, chemotaxis) do not select a direction because the c302 cell model cannot reproduce the atlas's inhibitory responses, and the c302 cell model produces no locomotor rhythm, so the motor neuron / muscle layer is replaced by a literature model (Boyle, Berri & Cohen 2012). See [What works and what does not](#what-works-and-what-does-not).

## How it works

```
 text ──► sentence embedding ──► protocol-mixture head ──► stimulus schedule (whitelisted neurons only)
 (MiniLM, frozen)              (small MLP, 267 phrases)    e.g. AVB 5 pA for 10 s

 stimulus ──► c302 C2 network in JAX (302 neurons, connectome fixed, fitted dynamics "Vfit")
          ──► AVB / AVA / SMD / RIV / RIS membrane potentials read out
          ──► Boyle 2012 bistable motor layer + stretch receptors (gated by AVB/AVA, steered by SMD, omega bend by RIV)
          ──► 2D viscoelastic rod on agar (resistive-force theory) ──► trajectory, curvature, wave frequency
```

Principles that every experiment follows (`PLAN.md` §1.1):

1. **The connectome is immutable.** What we fit are dynamical assumptions (synaptic thresholds, gains, leak), tracked as named variants (`V0`, `V1`, `Vfit`, …).
2. **The LLM is an experimenter, not a pilot.** It can only choose protocols from `worm/llm/protocols.py`, whose channels are sensory, command and steering interneurons. The whitelist is enforced by `validate()`.
3. **Every protocol cites a paper**, and every added mechanism outside the connectome (motor layer, steering, omega turn, RIS quiescence) is an explicit, flagged assumption with an ADR in `docs/DECISIONS.md`.
4. **Negative results are recorded** with the same care as positive ones.
5. Reported numbers are re-checked at dt = 0.05 ms in float64 (exploration runs use dt = 0.25).

## What works and what does not

| Layer | Result | Where |
|---|---|---|
| JAX re-implementation of c302 C2 | Matches NEURON to 0.001 mV RMS; runs faster than real time on a CPU | `docs/RESULTS_PHASE1.md` §1 |
| Comparison with the Randi et al. 2023 signal-propagation atlas | Dynamics beat anatomy (Spearman 0.13 vs 0.08, AUROC 0.65 vs 0.55 on 140 held-out stimulated neurons). Predictive power comes from gap-junction spread; chemical synapses are silent at rest. | §5–6 |
| Inhibition and sensory→command selection | **Negative.** 83 % of the atlas's inhibitory pairs have no synaptic path in the connectome; turning inhibition on destroys the predictive power and class-wise parameters (63 free) do not recover it. | §7 |
| Locomotion from c302 cells | **Negative.** No intrinsic oscillator, no wave from proprioceptive feedback, channel additions do not help. | `docs/RESULTS_PHASE2.md` |
| Boyle 2012 motor layer gated by connectome AVB/AVA | Forward 0.18 mm/s at 0.56 Hz, reverse 0.11 mm/s; literature 0.2 mm/s, 0.3–0.5 Hz | `docs/RESULTS_PHASE3_4.md` §1 |
| Steering, U-turn, local search, RIS quiescence | All 9 optogenetic-grade protocols produce the intended behavior | §1b |
| Sensory protocols (touch, ASH escape, chemotaxis) | **Negative.** No direction selection (same cause as inhibition above). | §2, §5 |
| Translator (embedding → protocol mixture) | Held-out phrase accuracy 0.86; on the behavior level 26/26 commands reach the intended movement axis | §3–4 |

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 (downloaded by uv). About 650 MB of reference data in `runs/` is not in git; see [Reference data](#reference-data).

```bash
uv sync                                                  # all dependencies (JAX, torch, NEURON, c302, sentence-transformers)
uv run python experiments/phase4_train_translator.py     # train the translator once (~1 min, CPU)
uv run uvicorn worm.server.app:app --port 8000           # open http://localhost:8000 and type a command
uv run pytest -q tests                                   # 19 tests, ~25 s
```

The web UI accepts Korean and English ("앞으로 가", "turn left", "make a U-turn", "춤춰봐", "watching YouTube"). It shows the worm, the stimulated neurons, the membrane potentials of the command/steering neurons and the protocol the translator chose with its reference paper.

### Reference data

Every experiment reads the NeuroML network exported by c302 from `runs/phase0/c302_C2_LW_Full_avb-ava/`. Either copy `runs/` from another machine (nothing else is needed) or regenerate it, which requires Java (jNeuroML) and a C compiler (NEURON `nrnivmodl`):

```bash
# macOS:  brew install openjdk@21        Linux:  sudo apt install build-essential openjdk-21-jre-headless
uv run python experiments/phase0_c302_reference.py --full     # several minutes, writes runs/phase0/
```

### Platform notes

- **Linux + NVIDIA GPU.** `uv sync` installs `jax[cuda12]` 0.7.1 and a CPU-only torch. The batched fitting scripts (`phase1c_fit.py`, `phase1d_fit.py`, `phase1b_sigprop.py`) run on the GPU: one fit iteration takes 2 s on an RTX 5090 versus 43 s on a 16-core CPU. Single-trajectory work (tests, web UI) is 3× faster on the CPU because kernel latency dominates, so the server and the tests pin `JAX_PLATFORMS=cpu`. When running several GPU jobs at once set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **macOS Intel.** Conditional dependencies keep JAX 0.4.38 and torch 2.2.2 (the last versions with Intel wheels). Phase 0 needs `brew install openjdk@21`; the script works around the broken libc++ headers of CommandLineTools when compiling NEURON mechanisms.
- **Apple Silicon.** Untested; the pins in `pyproject.toml` can be relaxed.
- The Python version is capped at 3.12 because torch 2.2.2 has no 3.13 wheels.

## Repository layout

```
worm/neural/     connectome.py (NeuroML parser), jaxsim.py (C2 simulator), variants.py (V0/V1/Vfit, θ), readers/
worm/body/       rod2d.py (2D rod), boyle_motor.py (Boyle 2012 motor layer), muscle_map.py
worm/sim.py      neural–body coupling (Worm), gates, steering, omega bend, behavior descriptors
worm/llm/        protocols.py (whitelist), phrases.py (training sentences), translator.py
worm/server/     FastAPI + WebSocket server;  web/index.html  the UI
experiments/     one script per experiment, phase0 … phase4
tests/           19 tests: NEURON equivalence, body model, motor layer, protocols, end-to-end
docs/            results, decisions, references (Korean)
runs/            outputs, not in git (~650 MB)
history/         raw Claude Code session transcripts (jsonl)
```

## Documentation

The documentation is written in Korean.

| Document | Content |
|---|---|
| [PLAN.md](PLAN.md) | Goals, principles, architecture, phase roadmap, risks |
| [docs/COMMANDS.md](docs/COMMANDS.md) | Command ↔ neuron-stimulation table with the supporting papers |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture decision records ADR-001 … ADR-016 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Phase 0 reference runs and the C2 model specification |
| [docs/RESULTS_PHASE1.md](docs/RESULTS_PHASE1.md) | JAX re-implementation, bistability, signal-propagation atlas, parameter fits, inhibition (negative) |
| [docs/BODY_MODEL.md](docs/BODY_MODEL.md) | 2D body model specification and validation |
| [docs/RESULTS_PHASE2.md](docs/RESULTS_PHASE2.md) | Neural–body coupling, proprioception, search for an oscillator (negative) |
| [docs/RESULTS_PHASE3_4.md](docs/RESULTS_PHASE3_4.md) | Protocols, steering / U-turn / quiescence, translator, UI, chemotaxis (negative) |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Papers, tools, datasets |
| [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | Draft paper outline |
| [docs/SESSION_LOG.md](docs/SESSION_LOG.md) | Work log and how to continue on another machine |
| [CLAUDE.md](CLAUDE.md) | Instructions read automatically by Claude Code |

## Key references

- Gleeson et al. 2018, *c302: a multiscale framework for modelling the nervous system of C. elegans* (the neuron model and connectome used here).
- Randi et al. 2023, *Neural signal propagation atlas of C. elegans*, Nature (the functional dataset the dynamics are compared with).
- Boyle, Berri & Cohen 2012, *Gait modulation in C. elegans: an integrated neuromechanical model*, Front. Comput. Neurosci. (the motor layer).
- Chalfie et al. 1985; Gray, Hill & Bargmann 2005; Turek et al. 2016 (touch circuit, steering / omega turn, RIS quiescence protocols).

The full list is in [docs/REFERENCES.md](docs/REFERENCES.md).

## Status

First project milestone closed on 2026-09-03. The project was developed with Claude Code; the complete session transcripts are in `history/`. Open questions are listed in [docs/SESSION_LOG.md](docs/SESSION_LOG.md) §4.
