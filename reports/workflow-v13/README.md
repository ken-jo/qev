# Workflow recovery after the consumed v12 final

The v12 cohort-policy candidate failed legacy text accuracy (89.70% versus the required 90%).
Both roadmap priorities and every original release gate remain mandatory. The selected v13
recovery passed all 14 exact-model checks and exact bundled-wheel text/image inference.
The HF package is prepared with 190 files and 189 checksum entries. No HF publication occurred.

[Delivery completion audit](release-completion.json) rechecks all 14 model gates, 174 acceptance
evidence fingerprints, 60 frozen sources, 60 public aggregates and the exact verified package.
Its 87 checks passed without new inference, training or changes to the release criteria.

- [Completed head recovery](readout-recovery/README.md): all eleven declared candidates failed.
- [Fresh validation and final data](fresh-data/README.md): complete, independently audited;
  both independent validations and the new final evaluation are complete.
- [First late-layer execution](backbone-recovery/README.md): allocation failure preserved.
- [Memory repair](backbone-memory-repair/README.md): the identical training pass and all three
  merged development evaluations completed; the 50% checkpoint was selected by the fixed rule.
- [Policy validation](policy-validation/README.md): admission and fitting passed; the frozen
  policy passed both independent populations and the original-group regression.
- [Fresh final evaluation](final-evaluation/README.md): runtime and final sources frozen;
  paired comparison, regressions, runtime and all 14 model acceptance checks passed.
- [Consumed-final uncertainty diagnosis](uncertainty-diagnosis/README.md): separate oracle
  ambiguity and additional model error, without evaluating any recovery candidate.
- [Recovery uncertainty limitations](uncertainty-diagnosis/recovery-final.md): same-request
  oracle decomposition, actual accepted risk, prior-shift intervals and calibration error.
- [Fixed original release gates](../../configs/workflow-release-v12.json).

Both completed finals are now regression/diagnostic evidence only. Recovery optimization and
selection used train/dev groups. Any future independent performance claim requires another
uninspected final population, with weights, policy and evaluation sources frozen beforehand.
