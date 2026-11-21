# CalcLM

A TI-84 Plus CE that writes its own programs, and a measurement of how often they are right.

CalcLM is a small language model that runs on a TI-84 Plus CE graphing calculator, on
unmodified hardware (Python edition, rev M, OS 5.8), with native code started through arTIfiCE
and the CE C libraries, and writes TI-BASIC programs. The calculator does all of the work
itself, with no computer attached.

What v2.6, the build on the author's calculator, does:

- **280 of 1,000 programs right** on the calculator's ten-formula topic menu, measured in the
  emulator on exactly the programs the calculator writes. The ten formulas were chosen because
  an earlier model could already write them.
- **Under 1% right on formulas absent from training.** On 20 held-out formulas, the training
  recipe v2.6 uses is right about 1 time in 400 (0.24%, averaged over 16 training runs,
  measured before its training data was cleaned).
- **About 0.67 tokens per second**, timed on a real calculator for this model shape; v2.6's own
  figure (0.669) is measured in the emulator.
- **Byte-identical to the PC reference** on every program compared on hardware: for v2.6, the
  two programs compared on 21 November.

It generalises to almost nothing beyond the formulas it saw often in training; the contribution
is the engine and the measurement.

**This is a research project, not a product.** There is no app to download or install. This
repository holds the paper and the code. It does not hold the trained model, and no program the
model wrote.

The paper is the full account:
[CalcLM: a ternary transformer that writes TI-BASIC on a TI-84 Plus CE](https://cwklurks.github.io/calclm/paper/calclm-paper.html),
on the project's GitHub Pages site. In this repository it is
[docs/paper/calclm-paper.html](docs/paper/calclm-paper.html); GitHub shows HTML files as source:
download it (Raw, then save) or clone the repository, and open the file in a browser. It covers
the whole project, every experiment, and has a plain-language view for readers without a
background in AI or statistics.

## What it is

- **The model.** A transformer with about 3.7 million weights in its main layers. Each weight
  is -1, 0 or +1, so the arithmetic becomes adding, subtracting or skipping. That is what lets
  it run on a 48 MHz eZ80 with about 150 KB of working memory.
- **On the calculator.** A C and eZ80-assembly program (`ce/calclm/`) runs the model. It
  matches the Python integer reference (`model/intref.py`) bit for bit.
- **Decoding.** An exact TI-BASIC state tracker (`tracker/`) masks impossible tokens, for
  example an `End` with nothing open. A linker decoder makes every `Goto` land on a real label.
- **Checked on hardware.** Every device release was checked seed for seed on a real
  calculator: it had to write the same program, byte for byte, as the reference on a PC.
- **The yardstick.** Quality is measured by running programs on the real TI-OS inside the CEmu
  emulator. A program passes a task only if it shows the right answer for three sets of inputs.

## What it found

The numbers come from the paper, as of 21 November 2026. The current release is v2.6, with
the topic menu, released on 21 November: the hardware-checked v2.6-rc2 bundle, unchanged. A
release here is the build installed on the author's calculator; it is not distributed. The
figures at the top are v2.6's.

The older program and speed figures below are v2.3-rc2's, an unreleased candidate whose model is
the best of 16 training runs of the best recipe that fits the calculator. It was picked after
the fact, so it flatters the recipe a little.

**Without a prompt** (what the calculator does on its own: a program on a random topic), of 300
programs about 1 in 3 run cleanly, finishing without an error or a hang and showing something.
About 1 in 5 have the shape of a math tool (they ask for numbers); without a prompt, nothing
checks whether they compute the right thing.

**With a prompt** (measured in the emulator). The model is given the opening of a program, a
title and the lines that ask for inputs, and writes the rest.
- On 40 practice formulas (circle area, Fahrenheit to Celsius and so on), about 7% of attempts
  are right (139 of 2,000).
- On 20 held-out formulas that no training step ever saw, 0.6% are right (6 of 1,000, all on
  one formula).
- Most passes come from a handful of formulas: 7 formulas give 90% of them.
- The project's milestone is 10%, averaged over training runs. The recipe's average is 4.7%,
  so the milestone has not been reached.

**Speed.** 0.67 tokens per second on a real calculator. Programs average about 16 minutes,
and nearly half run to the 1,024-token cap, about 25 minutes.

**What the experiments found:**
- **Correctness had to be measured directly.** An early score rose fifteen-fold. Then a blind
  audit found that none of 150 passing programs used the right formula. The task suite was
  built after that, and has been tightened twice since.
- **Some things helped:** real human-written math programs as training data, a wider model,
  and training four times longer.
- **Some things did not:** training eight times longer, distillation from a larger model, and
  drilling the model on practice tasks.
- **Verified extra programs did not teach new formulas.** These were programs written by
  another AI and checked in the emulator. They did not help on unseen formulas, and in larger
  doses they hurt.
- **A bigger model helps but does not fit.** A model with 8 layers instead of 6 scored 6.6%
  against 5.0% for its control. It does not fit the calculator's memory, file and speed limits,
  and attempts to shrink its memory needs have not closed the gap. On cleaned training data,
  the bigger model in its device format beat today's shape by 0.92 points, a marginal pass:
  the 95% interval runs from 0.02 to 1.81 points.
- **Cleaner training data cost no accuracy beyond a preset margin, but did not cut copying.**
  Dropping some sources, scrubbing credit and address patterns and removing duplicates left
  the score within its margin, and copying of real training text did not fall. The cleaned
  data was adopted for its cleanliness, not for a measured gain.
- **On the calculator's topic menu, picking the right training run matters as much as model
  size.** An exploratory sweep of 64 trained models (68,000 programs) found the bigger model's
  runs 3.3 points ahead of today's shape on average (24.7% against 21.4%). But on fresh
  programs the best bigger-model run, the best run of today's shape and v2.5-rc1's model tied
  (28.6%, 28.4% and 28.2%). So the bigger model will not be ported for now, and v2.6 (the
  build on the author's calculator, 21 November) uses the best run of today's shape. Its
  hardware check passed that day, byte-identical on the two programs compared.

## What is in this repository

| path | what |
|---|---|
| `docs/paper/` | the paper |
| `docs/index.html` | the GitHub Pages index, which links to the paper |
| `LICENSES/` | the CC BY 4.0 legal code |
| `model/` | training (ternary quantisation-aware training), the export format, the integer reference, the linker decoder, the topic menu |
| `ce/` | the calculator program (C inference core, eZ80 assembly, UI), the matvec kernels, CEmu benchmarks, build tools and tests |
| `tracker/` | the exact TI-BASIC program-state tracker (C, with a Python spec) behind the masks |
| `eval/` | the harness that runs programs on emulated TI-OS and classifies the outcomes |
| `lab/` | the measurement: task suites (`tasks.py`, `tasks_v2.py`, `tasks_v3.py`), judging, scoring, statistics, device cost, sampling, and the prompted decoder behind the topic menu (`e14.py`) |
| `corpus/` | the corpus pipeline (crawlers that respect robots.txt, extraction, cleaning, tokenizer), the token vocabulary and the corpus report with its licence survey (code only, no corpus) |
| `demo/` | a browser page that runs the calculator's own C core, compiled to WebAssembly |

Run and milestone names (E7, W, L4c-s11, M4) are defined in the paper's glossary (shapes,
recipes, seeds and milestones) and in its section on each experiment.

[docs/PUBLIC-EXPORT.md](docs/PUBLIC-EXPORT.md) describes what this tree leaves out and edits.
It is made from a private research repository, which keeps the full record.

## What is not published, and why

- **The trained weights and the calculator bundles.** v2.6's weights are not published. The
  model reproduces training text: a review of an earlier model's programs found credit lines
  that name real people, often minors when they wrote the programs, in 44 of 300, and the
  cleaned training recipe v2.6 uses still copies real training text in about 1 program in 200
  (0.54%). Many training sources also carry no licence or a GPL licence (`corpus/CORPUS.md`,
  "Licensing and terms"). A separate model trained only on licence-clean data is being
  evaluated as a possible downloadable release.
- **Any program the model generated.** The model sometimes reproduces training text nearly word
  for word, including credit lines with people's names.
- **The training data.** The corpus collected from ticalc.org, TI-Planet and GitHub, the mined
  seeds and the AI-written programs used in some experiments. It is other people's work and is
  never redistributed. The corpus report (`corpus/CORPUS.md`) describes the sources and their
  terms, without samples. Removal requests: an author who wants their programs left out of the
  corpus can open an issue on this repository's GitHub issues
  (https://github.com/cwklurks/calclm/issues).
- **The TI-OS ROM and emulator state.** The ROM cannot be redistributed.
- **The research record.** The experiment plans, write-ups, run records, hardware-check
  records, audit scripts, one-off experiment runners and working notes stay private. The paper
  reports their results and is the public record; where it cites them, it says so.

## Exams

Nothing installable is published, so there is no CalcLM to put on a calculator. Running it
needs a jailbreak (arTIfiCE) to start native code. And for any formula it can write, a
hand-written program stored on the calculator is faster and always right.

## What you can run

You can read everything, including how each number in the paper was produced. Without the
private files:

- **Runs:** the unit tests that need no corpus, model export, ROM or emulator (the others skip
  and say why); the integer reference and the training code on toy models (through their unit
  tests, `model/tests`); the tracker's host tests and the C core's host tests that need no model
  export (a C compiler is enough); and the task suites' judging rules, the decision rules and
  the statistics, on records you supply, as Python functions (`lab/tasks*.py`, `lab/stats.py`):
  the task modules' command lines judge through the emulator or read private run records.
- **Builds, with the CE C toolchain:** the calculator program (`ce/calclm/`, see its README for
  the prerequisites). It builds without a model; installing it needs model files, which are
  not published.
- **Does not run: judging programs.** It needs the CEmu emulator (GPL-3.0), a Python harness
  around it that is not published (`eval/emu.py` imports it as `tools.cemu` from the folder
  named by `CALCLM_CEMU_ROOT`), and a TI-84 Plus CE ROM of your own.
- **Does not run: training, the device checks and the browser demo.** Training needs the
  corpus; the device checks and the demo need the model exports. None of them is published.

The 20 held-out tasks are in `lab/tasks_v2.py`, so they are public: later work on them is no
longer blind to the outside world.

The corpus report's "Reproduce" recipe (`corpus/CORPUS.md`) can rebuild a corpus from the
original sites, but it has not been checked from a fresh clone.

Set up with [uv](https://docs.astral.sh/uv/) and run the tests, from the repository root:

```sh
uv sync
uv run python -m pytest -p no:cacheprovider -m "not slow"
```

- The project needs Python 3.14 or later (`requires-python` in `pyproject.toml`); uv fetches
  it if it is missing.
- On Linux, `uv sync` installs PyTorch with its CUDA libraries: about 3 GB to download and 5 GB
  on disk, although every test runs on the CPU.
- pip is not supported: `pip install .` fails, because the repository is not a package.
  Without uv, install the dependencies `pyproject.toml` lists (numpy, pillow, pytest, tivars
  and torch) by hand into a Python 3.14 environment, and run pytest from the repository root.
- Expect most tests to skip. A fresh clone gives about 1,380 passed, 2,360 skipped and 170
  deselected in a minute or two, and each skip says what it needs: almost always a private
  file, the ROM or the toolchain, otherwise a GPU or an opt-in setting.

## Licence

The code is released under the MIT licence; see [LICENSE](LICENSE). Copyright (c) 2026
cwklurks.

One exception: `ce/tools/shim.c` is licensed GPL-3.0-or-later because it adapts code from the
CEmu project; all other code is MIT. The GPL-3.0 text is at
https://www.gnu.org/licenses/gpl-3.0.txt.

The paper (`docs/paper/`) and the other prose in this repository (the READMEs and notes beside
the code) are released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/; legal code in
[LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)): anyone may share and adapt them if they
credit cwklurks.

CalcLM is by cwklurks.
