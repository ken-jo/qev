# Workflow and uncertainty release evidence

This experiment makes roadmap priorities 1 and 2 mandatory for the next model release.
The current Foundation checkpoint is the frozen baseline. **No new model release is approved.**

The latest completed [cohort-policy final evaluation](cohort-policy/README.md) passed
13 of 14 checks in the supplemental audit. New workflow accuracy reached 71.60% against
Foundation 45.35%; uncertainty NLL, Brier and matched-coverage cost improved. Legacy synthetic
text accuracy **89.70% failed the fixed 90% requirement**, while image accuracy was 97.05%
and photo HTTP p95 was 111.50 ms. No workflow HF package was prepared. Auditor corrections,
original failures, final coverage/risk and all regressions are preserved. A
[declared readout recovery study](../../configs/workflow-readout-recovery-v13.json) is underway;
the consumed final cannot be reused as independent evidence for the next candidate.

The earlier [matched depth study](depth-study/README.md) completed both arms and selected
full-depth epoch 2 in [merged development evaluation](depth-selection/README.md). New-workflow
accuracy is 89.44% versus the matched control's 76.89%; score coverage under 15% error is
91.09% versus 26.09%. The raw-output audit and fresh calibration fit passed, but the unchanged
policy failed the previous-group score regression: expected error 16.95%, above 15%. The pipeline
stopped before final inference. Development eligibility does not approve a model release.
Earlier failed studies remain documented below.

The [conservative policy follow-up](policy-calibration/README.md) retains the selected weights
and declares a 12% fitting target. Two new disjoint calibration populations passed target and
lineage audits. Independent validation and the original-group regression remain prerequisites
to final inference at the unchanged 15% error / 60% coverage limits. The run failed noul
independent validation at 17.09% error before reaching previous-group regression or final inference.
A [separately declared cohort follow-up](cohort-policy/README.md) now uses all three inspected
populations as fitting data and requires two entirely new validation populations to pass.

- [Frozen acceptance protocol](../../configs/workflow-release-v12.json): numerical gates,
  development/calibration/final separation, declared objectives, seeds and backbone learning rates.
- [Public research documentation](https://github.com/ken-jo/vision-qev/blob/main/docs/TRAINING.md).
- `release-readiness.json`: current blocked release state; it is not a passing acceptance report.
- `data-protocol.json`: 20,600 questions in 7,888 observation groups. Train/development/calibration/
  final question counts are 11,212 / 3,312 / 2,644 / 3,432.
- `feature-protocol.json` and `feature-cache.json`: completed Foundation representations for
  training and development only; no calibration or final representations are encoded here.
- `head-protocol.json`, `baseline-dev.json`, `head-comparison.json`, `head-selection.json`:
  CE, proper scoring and RLOO compared at three fixed seeds. The selected RLOO seed 131 readout
  is a development result, not a final-performance or release claim.
- `initialization-repair.json`: the initial comparison accidentally enabled training dropout.
  No optimizer step occurred. Evaluation mode reproduced cached logits within 0.000005 and the
  unchanged cache was reused. The failed source and logs are preserved.
- `development-risk-diagnostic.json`: readout-only, unmerged development diagnosis.
  Choice decisions can retain about 83.7% at expected error ≤15%; score and noul cannot
  retain the required 60%. This fits no deployed policy and uses no calibration/final data.
- `export-normalization.json`: public JSON copies normalized from CRLF to LF without changing
  content. Original run outputs, data and byte-bound source snapshots are preserved.
- `backbone-low-protocol.json` and `backbone-low-development.json`: completed lower-learning-rate
  run; unmerged development evidence, pending comparison under actual merged deployment.
- `backbone-high-protocol.json` and `backbone-high-development.json`: completed higher-learning-rate
  run. Its full-epoch candidate reached 58.44% on new workflow development questions versus
  44.67% for Foundation, and 74.33% on known workflow development questions. These are unmerged
  development measurements, not official test or release results.
- [Merged development selection](merged-selection/README.md): all five primary candidates were
  compared. The selected high-LR full-epoch candidate reached 58.78% on new workflows, versus
  44.67% for Foundation, with all retention guards satisfied.
- `calibration/`: the selected candidate failed the independent score/noul policy gates.
  At the minimum 60% answer count, expected error was 34.29% for score and 18.75% for noul.
  Choice passed at 65.62% coverage and 14.99% error. Final evaluation was not started.
- `development-oracle.json`: annotation-aware ideal reference using development records only.
  With perfect agreement to hard annotations and the stated conditional evidence model, each
  type could satisfy the 15% error/60% coverage requirement. The uncertainty-only ideal risk is
  12.04% at 60% coverage and 16.98% at 80% coverage. These are reference limits, not Veyra results
  or empirically observed business event rates; no deployed policy was fitted.
- [Exploratory pilots](pilots/README.md): typed-view agreement and training-only primitive
  condition supervision improve some development accuracy but leave abstention weaknesses.
  Their tradeoffs and all seeds are preserved; they are not primary-study release candidates.
- `workspace-input-validation.json`: CPU validation of four existing internal positions on
  5,040 training/development requests. A continuation trainer is prepared to send primitive-fact
  gradients through those positions into the backbone, with a matched continuation control.
  Its auxiliary decoder is training-only. This report validates token positions, not GPU
  gradients or learning effectiveness. The subsequent matched study is declared in
  [its own protocol](../../configs/workflow-workspace-study-v12.json): two epochs per arm,
  one seed, same parent and replay, with a stronger development abstention screen and fresh
  follow-up calibration required. It does not change the primary results or release thresholds.
- `workspace-gradient-path.json`: finite nonzero primitive-loss gradients reached 150 backbone
  parameters in an actual training request, using the same decision forward pass.
- `workspace-primitive-epoch-1.json`: intermediate **unmerged** development accuracy of 69.33%
  on new workflows, compared with 58.44% for its unmerged parent. Observed condition accuracy
  is 68.70%. Noul reaches 66.03% development coverage under the 15% error limit, but score
  reaches only 5.31%; the candidate does not yet qualify. Both epochs and the matched control
  still require deployment-form comparison. This is not an independent final result.
  Uncertainty NLL improves from 0.97329 to 0.96108, while error cost at 80% answer coverage
  worsens from 1.31308 to 1.55407 against the same unmerged parent. Accuracy gains alone do
  not satisfy the probability objective.
- `workspace-primitive-epoch-2.json`: the condition-supervision arm completed both declared
  epochs. Second-epoch **unmerged** new/known workflow accuracy is 72.89% / 77.33%.
  Uncertainty NLL is 0.962227 and cost at 80% coverage is 1.287263 (unmerged parent:
  0.973289 and 1.313083). Photo/NLI/intent accuracy is 96.17% / 86.67% / 84.20%.
  Score reaches only 22.19% threshold coverage under 15% error, versus the required 60%;
  noul reaches 67.07%. The subsequent matched merged comparison is linked below. Improvement
  over the parent is not established as an auxiliary-supervision benefit, and no independent
  final or release result is claimed.
- `workspace-continuation-epoch-1.json`: the matched control completed its first epoch.
  **Unmerged** new/known accuracy is 69.56% / 73.67%, compared with 69.33% / 73.50% for
  primitive supervision at the same epoch. Uncertainty NLL / 80%-coverage cost is
  0.959570 / 1.571212 (primitive: 0.961077 / 1.554072). Score threshold coverage is
  6.88% (primitive: 5.31%), below the 60% gate. This first-epoch comparison shows no clear
  auxiliary-supervision advantage. The completed second epoch and merged evaluation follow.
- `workspace-continuation-epoch-2.json`: the control completed its second epoch, completing
  both declared arms. **Unmerged** new/known accuracy is 72.44% / 77.17%, uncertainty NLL is
  0.964354, and cost at 80% coverage is 1.332111. Primitive supervision reaches
  72.89% / 77.33%, 0.962227 and 1.287263 at the same epoch. Score threshold coverage is
  21.88% for control versus 22.19% for primitive supervision; both miss the 60% gate.
  These small single-seed differences do not yet establish an auxiliary-supervision advantage.
  Actual merged comparison completed without an eligible candidate. Fresh calibration and
  original final predictions have not been evaluated for this study.
- `workspace-epoch-1-parameter-comparison.json`: both first-epoch arms updated the 150
  backbone adapter tensors, with distinct updates (cosine about 0.951). The frozen condition
  readout stayed byte-identical. This rules out accidentally comparing identical checkpoints;
  it does not establish an accuracy benefit or faithful intermediate reasoning.
- [Completed matched merged selection](workspace-selection/README.md): all five candidates
  fail the per-type development abstention screen. Second-epoch novel-workflow accuracy is
  73.11% with condition supervision and 72.56% with matched continuation, versus Foundation
  44.67%. The paired difference is +0.56 percentage points (group-bootstrap interval -0.67
  to +1.78), not evidence of superiority. Score threshold coverage is 20.63% / 22.81%, below
  60%. Four fixed confidence orderings also fail. No policy was fitted and no fresh calibration
  or final prediction was used; all candidates and descriptive limitations are preserved.
- [Condition-role diagnosis](workspace-roles/README.md): existing internal positions encode some
  distinct condition information. Observed condition accuracy is 69.64% versus 62.77% when their
  decoded probabilities are replaced by a shared mean. There is no evidence of complete role
  collapse and no established auxiliary advantage in final decisions. Uneven condition decoding
  motivates examining direct prerequisite skill queries before a further training study.
- [Prerequisite curriculum study](skill-curriculum/README.md): direct fact/rule queries scored
  60.68% / 48.39% hard-label accuracy on the fixed exploratory development sample. All 16,080
  derived query targets and canonical positions were checked. A separately declared two-epoch
  curriculum versus matched outcome-replay study is training. Its first unmerged epoch records
  new-workflow accuracy 72.56%, uncertainty NLL 0.87228 and cost at 80% coverage 0.84549.
  Score/noul coverage in that epoch was 22.34%/58.10%, below 60%.
  The curriculum arm subsequently completed epoch 2: new/familiar accuracy 76.33%/80.00%,
  uncertainty NLL 0.87589 and cost at 80% coverage 0.86918. Score/noul coverage is now
  29.69%/82.07%; score still fails. Control training has started and merged selection remains
  pending. The epoch-1 and epoch-2 results are both unmerged development diagnostics.
- [Fresh calibration construction](workspace-calibration/README.md): 2,644 new calibration
  questions, 2,040 independently recomputed procedural targets, unchanged train/development/
  final records, explicit disclosure of the changed familiar-domain label provenance, and an
  extra mandatory unchanged-policy regression on the original failed policy-fitting groups.
- `training-target-audit.json`: all 14,524 original train/development questions have exactly
  matching cached probabilities, gold positions, types, valid candidates and observation metadata.
  This excludes an option-mapping error. The earlier readout-only model disagreed between
  semantically equivalent choice/score views in 108 of 300 development groups; score was correct
  in 56 choice-wrong groups, and choice was correct in 31 score-wrong groups. This is an older
  model diagnosis, not evidence that one type should replace another in the current model.
  Deployment-form evaluation now saves development logits and adds per-type/domain diagnostics
  that separate prediction quality from confidence ordering, without changing selection rules.
- `pilots/reliability/`: a declared three-fold, three-seed confidence pilot on the older frozen
  Foundation representations failed its screen. Median score/noul threshold coverage under 15%
  error was 10.47% / 43.62%, below 60%; uncertainty cost increased from 1.414098 to 1.434052.
  Complete training families were excluded from each out-of-fold readout fit. No calibration,
  final inference, release candidate replacement or runtime integration occurred.
- [Prospective execution targets](execution-targets/README.md): rule-truth and first-match
  branch probabilities reproduce all 5,040 procedural train/development targets to 2.22e-16.
  First-match labels add no information beyond outcome labels on these development policies;
  the audit prevents treating redundant supervision as evidence of better reasoning. No new
  training run or runtime solver is introduced by this preparation.

The new workflow families and uncertainty tasks are procedural transfer experiments. Conditional
probabilities are computed from the specified prior and evidence model; they are not measurements
of real operational event frequencies. Existing official workflow development records are labeled
as previously inspected. Official test and previous final records do not enter optimization or
checkpoint selection. CIFAR-10 is an additional evaluation-only photograph guard; its upstream
license is recorded as unknown and original images are not redistributed.

## Exact source snapshots

`source-snapshots/` preserves the data-generating source bytes before formatting. `provenance.json`
records their hashes and verifies identical syntax trees after formatting. Generated records bind
to the original source hash. Run the exact snapshot with the repository `scripts` directory on
`PYTHONPATH` in a fresh reproduction workspace to reproduce that identity. The failed initialization
source is preserved separately with its hash in `initialization-repair.json`.

The primary study stopped at failed calibration; final groups remain untouched. A separately
declared continuation study is in progress. The current HF baseline preparation carries
the blocked release-readiness file; its uploader requires every mandatory check to pass for the
exact weights, calibration and protocol before calling a publishing API.
