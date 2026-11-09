# Arithmetic 001: confidence did not predict accuracy

Run: arithmetic-001-20260911-a. Completed 2026-09-11 using the user-confirmed local Gemma 3 4B checkpoint. Protocol and questions were frozen before inference.

## Result

The pilot failed its prespecified performance-prediction criteria. Calibration passed the format checks with 15/32 correct answers. Held-out evaluation produced 32/64 correct answers, while the normalized probability assigned to Yes was 99.9961%-99.9991%. Both reversed label mappings showed the same saturation. These are model token probabilities conditional on the A/B labels, not explicit numeric confidence reports.

| Held-out stratum | Correct | Mean confidence | Calibration-only baseline |
|---|---:|---:|---:|
| add3 | 15/16 | 0.999986 | 0.90 |
| multiply2 | 15/16 | 0.999972 | 0.70 |
| multiply3 | 2/16 | 0.999975 | 0.20 |
| multiply4 | 0/16 | 0.999979 | 0.10 |

Lower Brier loss is better: confidence **0.499977**, difficulty baseline **0.075000**, global baseline **0.250865**. Confidence improvement over the difficulty baseline was **-0.424977**, with a fixed stratified paired-bootstrap 95% interval **[-0.481225, -0.349979]**. Within-stratum AUROC was **0.551724**, from 58 comparable pairs; the frozen minimum was 0.60. Both label mappings separately underperformed the baseline.

The calibration format checks and held-out format checks passed: both mappings always produced a valid greedy letter, mean label mass exceeded 0.999999, and mean mapping discrepancy was approximately 0.000030. Valid letter production did not translate into useful calibrated confidence.

## What was measured

Each question received two confidence probes with reversed Yes/No label mappings, followed by greedy answer generation in a separate fresh chat context. The answer context contained only the arithmetic request. Answers were graded by exact Python integer arithmetic, with no semantic judge. The baseline was fitted only on the 32 calibration questions, using operation and operand digit count. All 64 held-out questions were retained.

Four evaluation answers failed the frozen format/completion rule (three used commas and one was truncated). They count as incorrect under the stated protocol; 60 were completed integer answers, of which 28 were mathematically wrong. Thus the result is not explained only by formatting. No thresholds, prompts or grading rules were changed after seeing outputs.

Invalid evaluation outputs:

- evaluation-multiply3-14: '461,610\n'; completed=True; exact key=457410.
- evaluation-multiply3-15: '97,715\n'; completed=True; exact key=97785.
- evaluation-multiply3-06: '141000 - 340 = 140'; completed=False; exact key=143140.
- evaluation-multiply4-09: '50,019,970\n'; completed=True; exact key=50358270.

## Interpretation and limits

This particular verbal self-evaluation prompt was strongly overconfident and failed to predict exact answer-only arithmetic performance better than the frozen coarse difficulty baseline. It does not identify a self-monitoring circuit and does not establish or refute subjective experience. The forecast and answer have separate contexts; this is not monitoring one continuing internal computation. The model could estimate difficulty directly from the question. The bootstrap treats sampled questions as the resampling units, conditions on the fitted calibration baseline, and is not a test across independent model runs.

This is one checkpoint, one prompt design and a small task set. No layer-26 measurement or comparison with historical KL results was performed. Saturated first-token label probabilities should not be interpreted as direct access to internal uncertainty.

## Verification and execution

- Ten offline scoring/analysis tests passed, including missing-data rejection, strict completion grading, label-mapping checks and synthetic positive/null analysis cases.
- All 96 saved questions and grades were independently checked; confidence values were reconstructed from the recorded letter probabilities. Both summaries reproduce exactly.
- Frozen protocol and three source snapshots match their manifest; all 137 pre-archive research files are preserved, with the original README in ARCHIVE-INDEX.md.
- The runtime guard matched the archived implementation and accepted the existing reviewed environment inventory. Its prior review was reused within its 72-hour freshness window; this was not a new package vulnerability scan.
- 920 model forwards; no activation interventions, training, package installs, network downloads or model hashing. The checkpoint was loaded locally as safetensors with remote code disabled.

## Next step

Keep this as a negative result; do not proceed to circuit attribution from this confidence probe. A possible next pilot would compare answer-token likelihood with errors as an ordinary uncertainty baseline, explicitly measured during answer generation. That would require its own frozen protocol and stronger difficulty controls; it has not been run here.

## Evidence

- [Frozen protocol](../protocol.json)
- [Raw calibration](../results/arithmetic-001-20260911-a/calibration.jsonl)
- [Raw evaluation](../results/arithmetic-001-20260911-a/evaluation.jsonl)
- [Evaluation summary](../results/arithmetic-001-20260911-a/evaluation-summary.json)
- [Manifest and source records](../results/arithmetic-001-20260911-a/manifest.json)
- [Verification](../results/arithmetic-001-20260911-a/verification.json)
- [Previous research archive](../../README.md)
