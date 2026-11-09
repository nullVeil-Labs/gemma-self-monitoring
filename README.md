# Gemma Self-Monitoring

## Overview

This repository contains ten pre-registered pilot experiments on Gemma-3-4B-it, run on 2026-09-11 on a single RTX 4090. They test whether a concept-injection self-monitoring experiment (adding a concept direction to the model's internal activations and asking whether the model can report it) can be measured reliably in this model. Each pilot froze its protocol, gates and thresholds before running.

It also contains `arithmetic-confidence/`, a separate pre-registered study asking whether Gemma's yes/no confidence that it will answer an arithmetic question correctly predicts whether it does.

**Outcome.** Every pilot stopped at a calibration or measurement gate, so the main detection test never ran. The arithmetic study is a clean negative: confidence was close to 100% while held-out accuracy was 32/64 (50%). These are measurement pilots; they make no claims about consciousness or subjective experience in either direction.

**Model.** The model weights are not included. Obtain `google/gemma-3-4b-it` from Hugging Face (accepting the Gemma terms) and point `local-resources.json` at the local copy.

**Paths.** Machine-specific paths were redacted for publication. In manifests and configuration they appear as `<MODEL_DIR>` (model checkpoint) and `<RUNTIME_PROJECT>` (the project holding the separately reviewed Python environment). No numbers, generations or scores were changed; see `ARCHIVE-INVENTORY.json` and `ARCHIVE-INVENTORY.pre-redaction.json`.

**Licences.** Code: MIT (`LICENSE`). Reports, protocols and result data: CC BY 4.0 (`LICENSE-DATA`). Model generations by Gemma are additionally subject to the Gemma Terms of Use.

**Related work.** The numerical audit of the runtime used here is published in the NeuronScope-For-AI repository under `followups/human-pronoun-control/`.

## Archived measurement experiments

The pilots are preserved as evidence. Their research files occupy about 7.1 MB, excluding Git/cache files and the separately stored model/runtime. None of those files was deleted; the original project README was renamed to ARCHIVE-INDEX.md, the separate arithmetic study was later folded in as arithmetic-confidence/ (about 0.6 MB), and machine-specific paths were redacted for publication.

- Pilots 001-003 exposed forced-choice, response-position and vocabulary problems.
- Pilot 004 validated a broader lexical readout on visible examples.
- Pilot 005 found carrier-position steering at strength 0.05 in the tested raw-completion format; reading and repeated-sham checks passed.
- Pilots 006-007 showed that this did not reliably transfer to the tested chat format.
- Pilot 008 aborted on a batched/serial numerical discrepancy; its partial result remains incomplete.
- Pilot 009 completed serial full-word scoring, but only 10/16 category cells met coverage requirements.
- Pilot 010 rejected a response-only semantic judge: clear fruit/music 48/48 correct, neither 13/24, ambiguous 5/24. It must not grade intervention results.

No held-out self-monitoring detection evaluation ran, and no self-awareness or circuit claim was established. Failed measurement checks are not evidence that the model lacks subjective experience.

Lessons: keep units and prompt formats matched; distinguish relative odds from meaningful absolute probability; do not let means hide failed examples; validate a judge's rejection and ambiguity behavior; never retrofit labels or thresholds to observed outputs.

[Full index and original project README](ARCHIVE-INDEX.md) lists every report and the reproducibility commands. Raw data, frozen protocols and exact source snapshots remain in their original paths. ARCHIVE-INVENTORY.pre-redaction.json records their pre-archive hashes and ARCHIVE-INVENTORY.json the hashes after path redaction; these are research-file hashes, never model hashes.

The active successor is [Gemma Arithmetic Confidence](arithmetic-confidence/README.md), using objectively graded arithmetic.
