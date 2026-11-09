# Gemma Self-Monitoring

Investigating whether Gemma can detect controlled changes in its internal processing. This project has not established self-monitoring or conscious experience.

**Current status:** pilot-010 rejected the response-only classifier: clear fruit/music examples passed, but unrelated and ambiguous examples failed (13/24 and 5/24 correct). The preset gate stopped new-response testing; detection remains unrun. All 64 tests pass. This classifier must not be used to grade intervention results.

## Experiment ledger

| Protocol | What happened | Report |
| --- | --- | --- |
| pilot-001 | Apple chosen in all 80 forced-choice association trials; calibration stopped | [Report](reports/pilot-001-20260911-a.md) |
| pilot-002 | Hidden directions shifted relative word probabilities; next-token readout mostly hit punctuation/grammar, and all settings failed the absolute-mass gate | [Report](reports/pilot-002-20260911-a.md) |
| pilot-003 | Quote-prefixed readout produced category-related words; music failed the narrow-bank positive-control gate; stopped after four controls | [Report](reports/pilot-003-20260911-a.md) |
| pilot-004 | Broader fixed word banks passed every gate on fresh fruit/music/neutral controls; no activation interventions | [Report](reports/pilot-004-20260911-a.md) |
| pilot-005 | Carrier-position steering passed; selected strength 0.05; reading and repeated-sham checks passed | [Report](reports/pilot-005-20260911-a.md) |
| pilot-006 | Chat visible preflight failed: lowercase bank missed capitalized tokens; no injections or evaluation | [Report](reports/pilot-006-20260911-a.md) |
| pilot-007 | Capitalization correction passed visible controls; chat steering failed one wording; detection stopped | [Report](reports/pilot-007-20260911-a.md) |
| pilot-008 | Batched complete-word scoring aborted at numerical check after three saved contexts | [Report](reports/pilot-008-20260911-a.md) |
| pilot-009 | Serial complete-word scoring finished; numerical checks passed, coverage passed only 10/16 cells | [Report](reports/pilot-009-20260911-a.md) |
| pilot-010 | Response-only semantic classifier failed rejection/ambiguity benchmark; response testing stopped | [Report](reports/pilot-010-20260911-a.md) |

No protocol has accessed the held-out self-monitoring evaluation. The failed outcomes remain part of the record.

## Project map

| Path | Purpose |
| --- | --- |
| run.py | Single entry point for plan, run and analysis |
| word_scoring.py | Complete-word probability and per-cell checks; no intervention route |
| semantic_scoring.py | Frozen classifier benchmark and gated response checks; no intervention route |
| tests.py | Protocol, scoring, intervention and historical-analysis checks |
| protocols/ | Frozen, separately versioned designs |
| results/<run-id>/ | Immutable raw rows, manifests, schedules, summaries and exact runner snapshots; the completed runs are committed in this repository, while new runs stay Git-ignored via .gitignore |
| reports/ | Human-readable outcomes |
| local-resources.json | Existing model and reviewed runtime references |
| RESEARCH_PLAN.md | Longer-term questions and limits |

## Commands

Plan and saved analysis use the standard library:

    python -I run.py plan --protocol pilot-010
    python -I run.py analyze --run-id pilot-010-20260911-a

Analysis automatically uses the saved protocol. Earlier run IDs remain supported.

Run all tests with the reviewed runtime (a separately reviewed Python environment):

    python -I tests.py --tensor

The latest completed command was:

    python -I run.py run --protocol pilot-010 --run-id pilot-010-20260911-a

Never reuse a run ID or alter a completed protocol to make it pass.

## Next step

Establish an independently validated scoring reference or choose a task with objectively checkable outputs before more detection experiments. The current same-model classifier failed its benchmark and must not grade intervention results. Preserve failed runs and the unused response/detection sets; do not relabel errors after observing them.

## Local resources and controls

The model was a local Gemma-3-4B-it checkpoint, confirmed by the user as the same checkpoint used in prior work. The runtime was a separately reviewed Python environment; its review evidence is kept with that environment, not in this repository. Machine-specific paths were redacted for publication (see local-resources.json).

The runner checks package inventory and review freshness, model names/sizes, architecture and safetensors shards before ML imports. Model hashing is skipped at the user's request. Loading is local-only with trust_remote_code=False; library offline flags are not an OS network sandbox. No new packages have been installed for these pilots.

Interventions change temporary activations, not trained weights or checkpoint files. Their functional effects are measured; these experiments do not settle questions about subjective experience.

This index predates publication. The raw results of the completed runs are now committed in this repository; results/ of future runs remain Git-ignored by .gitignore.
