# What this public export leaves out

This repository is a public export of CalcLM's private research repository, made by
`tools/public_export.py` from commit `5d93c39ec9b1`. CalcLM publishes its paper and its
code. It does not publish v2.6's trained weights (the README says why; a separate
model trained only on licence-clean data is being evaluated as a possible downloadable
release), programs the model generated, the training data, the TI-OS ROM or emulator
state, and the research record behind the paper (plans, write-ups, run records,
working notes) stays private: the paper is the public record. The export publishes
only files its allow-list names, leaves out or edits the files described below, and
was scanned for generated programs, token-id arrays, corpus samples, binaries,
personal e-mail addresses and private names.

## Left out (670 files)

By category, with the export's rules:

- plans (`PLAN-*.md`, `PHASE*.md`): 27 files. Written and frozen before each experiment; the paper reports what they fixed.
- write-ups and research notes (`lab/*.md`, `docs/research/*`): 33 files. The paper is the public record of their results.
- experiment runners (`lab/e1_*.py`, `lab/e1[0-35-9]*.py`, `lab/e14_*.py`, `lab/e[2-9]*.py`, `lab/g[0-9]*.py`, `lab/e18_taxonomy.json`, `lab/tests/test_e1_*.py`, `lab/tests/test_e1[0-35-9]*.py`, `lab/tests/test_e14_*.py`, `lab/tests/test_e[2-9]*.py`, `lab/tests/test_g[0-9]*.py`, `lab/tests/e24/*`, `runs/*`): 236 files. One-off code for a single experiment, its tests and the job, service and timer files that ran it.
- experiment-specific lab pipelines (`lab/reward.py`, `lab/expert_iter.py`, `lab/task_variants.py`, `lab/train_tasks.py`, `lab/breadth*`, `lab/judge_pool*`, `lab/rejudge.py`, `lab/tests/test_reward.py`, `lab/tests/test_expert_iter*.py`, `lab/tests/test_task_variants.py`, `lab/tests/test_train_tasks.py`, `lab/tests/test_breadth*.py`, `lab/tests/test_judge_pool*.py`, `lab/tests/test_rejudge.py`): 26 files. E9's reward and expert iteration, the E13 and E18 breadth data and the multi-host judge pool, whose code imports experiment runners.
- audit scripts (`scripts/*`): 11 files. The blind audits' drawing and labelling scripts.
- earlier work (`experiments/*`, `evo/*`, `viz/*`, `ANGLES*.md`, `FINDINGS.md`, `RESULTS.md`, `SPEC.md`, `VISION.md`, `prompt-*.txt`): 237 files. Exploratory scripts, the first specification, the vision and findings notes, novelty-check prompts, the evolution harness and the dashboard.
- hardware-check records (`hw/checks/*`): 17 files. Programs received from the calculator, photos of its screen and the check sheets.
- the calculator log (`hw/DEVICE.md`): 1 file. Versions, free memory and installs of unpublished bundles on the author's calculator.
- install guides (`ce/calclm/INSTALL*.md`): 4 files. For calculator bundles that were never published.
- a toolchain example (`ce/hello/*`): 3 files. A byte-identical copy of the CE C toolchain's hello-world example, under the toolchain's licence, not CalcLM's; nothing uses it.
- export and release tooling (`tools/*`, `.githooks/*`, `ce/tests/test_bench_import.py`): 25 files. This export's scripts, the release guard, the pre-commit hook, and a test that the private tools package and the CEmu wrapper coexist.
- working notes (`docs/agents/*`): 49 files. Status, log, task specs and reports, audits, process proposals, release notes and the publishing runbook.
- the public README's source (`README.public.md`): 1 file. Published as README.md.

## Edited (19 files)

- `ce/BENCH.md`: names the CEmu harness, not its private repository.
- `ce/calclm/README.md`: release sections marked as records of unpublished releases; the build's prerequisites noted; links to files that are not published turned into plain text.
- `ce/tests/test_calclm_kv4_build.py`: the check of a spike worktree's links to the author's main checkout is left out.
- `ce/tests/test_calclm_ld2.py`: the hardware-seed check, which reads an unpublished hardware-check record, is left out.
- `ce/tests/test_calclm_v2.py`: the INSTALL.md check skips without the unpublished install guide.
- `ce/tools/bench.py`: comments name the CEmu harness, not its private repository.
- `ce/tools/build.sh`: no default CEmu core folder: CALCLM_CEMU_ROOT or CEMU_CORE names it.
- `corpus/clean.py`: an error message cites no unpublished plan.
- `corpus/extract.py`: no default folders of the author's own programs: CALCLM_EVAL_DIRS names them.
- `demo/README.md`: links to files that are not published turned into plain text.
- `docs/paper/calclm-paper.html`: citations of files that are not published named in words; byline credits cwklurks.
- `eval/EVAL.md`: names the CEmu harness, not its private repository.
- `eval/emu.py`: no default CEmu harness checkout: CALCLM_CEMU_ROOT names it.
- `lab/emulator_lock.py`: the lock file is under this checkout's runs/ instead of the author's main checkout.
- `lab/synth.py`: a comment names no unpublished tool.
- `lab/tests/test_emulator_lock.py`: checks the lock path above.
- `lab/tests/test_function.py`: docstring describes the hand-written fixture.
- `model/kv4qat_train.py`: error messages cite no unpublished plan.
- `model/tests/test_export.py`: the torch-free worker check leaves out lab/e1_nll.py, an experiment runner.

## Replaced or added (4 files)

- `.gitignore`: the private ignore rules replaced by the public ones (data, runs, environments, build outputs, programs, weights and ROMs).
- `README.md`: the private repository's README replaced by the public one.
- `lab/tests/function_cases.json`: 26 model samples and their emulator screens replaced by hand-written programs and screens (made by lab/tests/make_function_cases.py), so lab/tests/test_function.py still runs.
- `lab/tests/make_function_cases.py`: added: the script that writes the hand-written lab/tests/function_cases.json (tools/public/ in the private repository).

## Names, paths and time zones

In 19 published files, the author's first name became
"the author" and the full name the handle cwklurks, the repository's path
`<repo>`, other home paths `~/`, and the time zone "local time"; a local rating
model is named generically. A published file that records the sha256 of a file
changed here records the changed file's sha256 instead ("sha256 pins").
Replacements: names 239, home paths 5, sha256 pins 2.

## Mentioned but not published (103 files)

103 published files mention left-out files by path, in comments,
docstrings or prose. The mentions are left as they are; those files exist only in
the private research record. By what they mention:

- plans: mentioned in 56 files
- write-ups and research notes: mentioned in 8 files
- experiment runners: mentioned in 1 file
- earlier work: mentioned in 22 files
- hardware-check records: mentioned in 1 file
- install guides: mentioned in 2 files
- working notes: mentioned in 33 files

## Reproducibility

Judging programs needs the CEmu emulator (GPL-3.0), a private harness around it and a
TI-84 Plus CE ROM, none of which is published; training needs the corpus, which is
not published either. See the README for what can be run from this repository alone.
