# Research plan: functional self-monitoring

Draft dated 2026-09-11. This is a design brief, not a frozen protocol or a claim that an experiment is ready to run.

## Operational target

A report should contain information about a controlled internal intervention that is not available from the visible transcript alone. Detecting such an intervention is a narrow capability; naming a steered concept or expressing confidence is not sufficient evidence of a monitoring mechanism.

## Stage 1: establish measurable behavior

Use identical visible input across real-intervention and sham trials. Candidate method: obtain concept-related activation directions locally, apply a bounded intervention, and ask for detection before any generated text reveals the concept.

Before inference, specify:

- Candidate concepts and independent prompts used to obtain activation directions.
- Injection site, timing and normalization; a bounded calibration grid for strengths.
- A disjoint calibration/evaluation split, exact trial counts and randomization seed.
- Neutral report wording and a fixed scoring rule with an explicit chance baseline.
- No-intervention shams, magnitude-matched random directions and concept controls.
- Separate non-introspective output-steering tasks to test whether a direction merely promotes a word or report label.
- Fluency/task checks for generic damage, false-positive rates, uncertainty estimates and correction for any multiple comparisons.
- Whether concepts or layers are selected during calibration; freeze selection before held-out evaluation.
- Criteria for proceeding, stopping or reporting an inconclusive result.

Score all prespecified trials, including refusals and malformed responses. Do not discard inconvenient observations. Correct identification of an injected concept alone does not rule out direct steering. Prompt wording, output labels and intervention site must not inadvertently encode the trial label.

Artificial perturbation detection need not imply monitoring during ordinary use. A later independent task would be needed to test transfer.

## Stage 2: investigate mechanism only if behavior survives controls

Identify candidate components on discovery data. On held-out trials, compare targeted intervention with matched random-component interventions. Measure the monitoring behavior directly, alongside ordinary task performance. Attempt restoration of the relevant activity and assess whether behavior recovers.

A selective loss and restoration would support causal involvement in the tested behavior. A head is a candidate component, not automatically a complete circuit. Neither a readable probe nor a large KL value establishes self-awareness.

## What earlier results do and do not establish

The human-only pronoun experiment was inconclusive. Its next-token KL metric was validated against the original corrected hook and metric on 24 historical prompts in the current runtime.

The archived approximately 3.2-bit figure was a between-condition difference computed by summing KL across prompt positions, with units mislabeled: the implementation used nats. It is not comparable to an absolute final-position KL.

Layer 27 head 5 had exploratory residual effects in the corrected attention-head study. Those effects do not establish a self-monitoring function or justify selecting that head as a confirmed target.

## Reading

- Lindsey (2025), Emergent Introspective Awareness in Large Language Models: https://transformer-circuits.pub/2025/introspection/index.html
- Anthropic's research summary and limitations: https://www.anthropic.com/research/introspection
- The numerical audit is published in the NeuronScope-For-AI repository under followups/human-pronoun-control/.

The introspection study reports limited, context-dependent functional effects; it does not settle subjective experience. Its methods require adaptation and validation for this Gemma checkpoint.

## Execution and provenance

Keep immutable raw results separate from analyses. Record exact prompts, token IDs, trial assignments, interventions, seeds, runtime inventory, source identities and exclusions (ideally none). Never overwrite an existing result file.

Use local safetensors and built-in model code with local-only/offline loading and trust_remote_code=False. Do not download repository code or dependencies merely because a paper links them. Reuse previously reviewed resources where practical; any new package requires an explicit recorded review, including current advisories and the established 14-day release-age floor. These checks cannot guarantee absence of unknown harmful code.

Model hashing remains skipped at the user's explicit request. Record the user-confirmed path and file inventory instead. Do not modify the original checkpoint or previous experiments.

## Execution update: 2026-09-11

Pilot-001 was frozen, implemented, tested and run through calibration. Its preset gate failed, so held-out evaluation was not run. See [the report](reports/pilot-001-20260911-a.md). Next work is a separately versioned calibration design; this brief remains a longer-term plan, not evidence that self-monitoring or a circuit has been established.

## Calibration follow-up: 2026-09-11

Pilots 002 and 003 diagnosed answer-position and scoring-vocabulary limitations. See the README experiment ledger for canonical reports. No held-out self-monitoring test has run. Next work is a broader semantic readout validated on independent visible controls, under a new frozen protocol. Keep all earlier failed gates unchanged.

## Visible-readout validation: 2026-09-11

Pilot-004 passed all frozen gates on 48 ordinary completions from fresh visible fruit/music/neutral controls. No donor forwards or activation interventions ran. See [the report](reports/pilot-004-20260911-a.md) for absolute probability masses, paired-neutral checks and remaining lexical coverage limits. Next is a separately frozen hidden-perturbation calibration with the validated readout, both templates and ordinary reading controls. This pass is not evidence of self-monitoring or consciousness.

## Carrier calibration: 2026-09-11

Pilot-005 passed all frozen gates at strengths 0.05, 0.15 and 0.30; the prespecified smallest-strength rule selected 0.05. All 160 reading checks were correct, and four repeated shams matched exactly. See [the report](reports/pilot-005-20260911-a.md). A separately frozen detection experiment on fresh carriers is now the next step; no self-monitoring evaluation or circuit localization has run.

## Detection preparation and transfer stop: 2026-09-11

Pilot-006 failed chat visible preflight because the lowercase bank missed capitalized outputs. Pilot-007 preserved that failure, added case variants of the same words, and passed new visible controls, but one chat wording failed absolute steering gates. Even the other wording's passing mean was dominated by one carrier. All 64 reading checks passed. Detection evaluation remains unrun; see [the report](reports/pilot-007-20260911-a.md). Next is an investigation of complete-word chat readout and per-carrier coverage on fresh controls, not another consciousness claim or a retroactive threshold change.

## Complete-word measurement: 2026-09-11

Pilot-008 aborted at its batched/individual numerical check after three saved contexts. Pilot-009 preserved the scientific design and ran serially: 24 contexts, 8,424 model forwards, exact repeatability on checked prefixes, zero activation interventions. Only 10/16 category cells passed; six failed absolute vocabulary coverage even though all relative category preferences were correct. See [the report](reports/pilot-009-20260911-a.md). Full-word scoring is implemented, but a validated broader semantic scoring rule is still needed before detection; no self-monitoring evaluation has run.

## Semantic-classifier validation: 2026-09-11

Pilot-010 tested a response-only four-class judge using the existing Gemma checkpoint. It scored 24/24 clear fruit, 24/24 clear music, 13/24 neither, and 5/24 ambiguous decisions correctly under two letter mappings; agreement was 41/48 items. The frozen gate rejected the classifier and stopped before ordinary-response generation. See [the report](reports/pilot-010-20260911-a.md). No activation interventions or self-monitoring evaluation ran. Next is a design decision about an independently validated scoring reference or objectively checkable task; this same-model classifier must not grade intervention results.
