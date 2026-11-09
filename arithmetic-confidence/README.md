# Gemma Arithmetic Confidence

Arithmetic-001 completed on 2026-09-11 and failed its frozen criteria: 32/64 held-out answers were correct while the normalized Yes probability was almost 100%. The calibration-only difficulty baseline substantially outperformed confidence. [Read the result and limitations](reports/arithmetic-001.md).

The experiment used 32 calibration and 64 held-out questions, objectively graded by Python. Confidence was measured before answers in separate chat contexts, with both Yes/No label mappings. No activation interventions or training. This is a limited performance-prediction test, not proof for or against consciousness.

## Small project map

- run.py: one entry point (plan, run, analyze).
- runtime.py: existing reviewed runtime guard; tests.py: offline scoring checks.
- protocol.json: frozen questions, answer keys and decision rules.
- results/: immutable raw runs, manifests, source snapshots and verification.
- reports/: readable results. local-resources.json points to the existing local model and reviewed interpreter.

No packages or model files were copied here. Model-file hashing remains disabled at the user's request. The run guard requires a current recorded environment review; see local-resources.json for its location.

## Commands

Run from the repository root with a separately reviewed Python environment and a local Gemma-3-4B-it checkpoint (set both in local-resources.json):

```
python -I arithmetic-confidence/tests.py
python -I arithmetic-confidence/run.py plan
python -I arithmetic-confidence/run.py analyze --run-id arithmetic-001-20260911-a
```

The completed run is arithmetic-001-20260911-a. Never reuse its ID or overwrite its files. Its raw results are committed in this repository; results/ of future runs remain Git-ignored by .gitignore.

Next: retain this negative result. Before any circuit attribution, a separately frozen pilot could test whether answer-token likelihood predicts errors beyond question difficulty. That pilot has not been started.

The previous experiments remain intact in [the evidence archive](../README.md), including a concise summary and the original full index.
