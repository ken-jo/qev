# Direct prerequisite curriculum: declared study in progress

**Both arms completed both declared epochs. The five-candidate merged evaluation failed the
score coverage requirement for every candidate.** See the [complete merged results and paired
comparison](../curriculum-selection/README.md). The sections below preserve the chronological
unmerged measurements and study procedure; none constitutes a release pass.
The previous matched workspace study failed its score abstention development screen. Its
condition positions encode some distinct information, but condition decoding is uneven and
condition supervision did not establish a decision advantage over matched continuation.

## Prospective prerequisite diagnosis

Before the new training, 1,200 derived development queries were fixed: one fact and one rule
condition per each of 600 original states. The preceding condition-supervision epoch-2 checkpoint
was measured with merged BF16 weights and its existing prompt/calibration:

| Query | Questions | Questions with a definite target | Hard-label accuracy | NLL |
|---|---:|---:|---:|---:|
| Individual condition | 600 | 412 | 60.68% | 0.72526 |
| Rule condition, independent of priority | 600 | 372 | 48.39% | 0.95130 |

These are new query forms with randomized proposition polarity, choice keys, ordinal rubric
direction and question type. The figures mix reasoning, question interpretation and rubric
binding; they do not isolate numeric comparison errors. They are exploratory development
diagnostics, not a new independent benchmark or release criterion. All type breakdowns are
preserved in `diagnostic-results.json`.

## Data and label integrity

The dataset contains 12,480 training skill queries from 960 original training groups and 3,600
development skill queries from 300 development groups. Each original state supplies four fact
queries and one query per rule condition. The state text stays byte-identical; no intermediate
answer is inserted into the input. Derived views retain the source group, split and family.

Targets use the stated priors and sensor model and the independently audited 16-world execution
annotations. A separate parser re-read the rendered propositions and criterion directions for
all 16,080 queries, with maximum target difference 2.22e-16, and checked canonical training
positions. That small roundoff at one was clamped to the probability range; larger errors reject
the build. The audit shares the previously checked primitive/Boolean parsers and does not
establish real business event frequencies. Calibration/final states were not interpreted.

## Frozen matched study

The [study declaration](../../../configs/workflow-curriculum-study-v12.json) was saved before
training. Both arms start from the same prior epoch-2 checkpoint, seed 197, two epochs,
backbone learning rate 3e-5 and the existing RLOO+CE objective.

- Every epoch includes all 11,212 original training questions and unchanged Foundation replay.
- The curriculum arm adds 3,840 prerequisite questions: one fact and one rule per each original
  complete or uncertain training state.
- The control adds 3,840 original outcome questions on exactly those same states.
- Both match state order, forward/backward counts, optimizer steps and schedule. Question
  wording/type/candidate count differ, so exact token counts and FLOPs are not matched.
- No auxiliary decoder, inference module, runtime prompt change or generated answer token is added.

Parent plus both epochs of both arms form the entire candidate pool. Selection uses original
development questions, with the unchanged retention/probability criteria and each type's
60% minimum coverage under 15% error. Skill accuracy cannot override this screen. One training
seed remains a limitation; improvements require actual merged evaluation.

Only a qualifying candidate can proceed to the still-uninspected fresh calibration design,
unchanged-policy regression on the original failed calibration groups, frozen final evaluation
and all original accuracy/runtime/HF package requirements. No gate is lowered.

## First curriculum epoch: interim development evidence

The first epoch completed. Both columns below use unmerged development inference; the parent is
the preceding condition-supervision epoch-2 checkpoint. The later merged comparison will govern
selection. These are repeatedly inspected development groups, not independent final results.

| Development metric | Parent | Curriculum epoch 1 |
|---|---:|---:|
| New-workflow accuracy | 72.89% | 72.56% |
| Familiar-workflow accuracy | 77.33% | 76.33% |
| Uncertainty NLL | 0.96223 | 0.87228 |
| Uncertainty error cost at 80% coverage | 1.28726 | 0.84549 |
| Choice coverage at at most 15% error | 95.89% | 96.32% |
| Score coverage at at most 15% error | 22.19% | 22.34% |
| Noul coverage at at most 15% error | 67.07% | 58.10% |

Probability fit and matched-coverage cost improved, but neither score nor noul reaches the
mandatory 60% coverage. The recorded fact/rule hard-label diagnostic accuracies are 67.72% and
68.01%. Their pretraining measurements above used merged weights; matched deployment-form
comparisons are still pending. Auxiliary task gains cannot establish final decision gains.
Photo, NLI and intent development accuracies are 96.50%, 86.67% and 84.42% respectively.
The declared study continued after this interim result. No curriculum advantage, calibration
pass or release approval is established by this first result.

`skill_curriculum-epoch-1.json` contains all aggregate metrics and checkpoint/source identities.
`scripts/export_curriculum_progress.py` verifies completed checkpoint identities and exports
immutable epoch summaries. It does not run model inference or alter candidate selection.

### Where the first-epoch probability changes occurred

Each condition has 100 development questions. The following comparison uses the same unmerged
parent and first-epoch checkpoints. Error and cost use 80% coverage within each condition, so
these accepted subsets differ from the combined uncertainty population's 80% coverage.

| Evidence condition | Parent → epoch-1 NLL | Parent → epoch-1 expected error | Parent → epoch-1 cost |
|---|---:|---:|---:|
| Conflicting | 0.96241 → 0.93902 | 47.19% → 48.39% | 1.49614 → 1.05594 |
| Missing | 0.99212 → 0.89326 | 47.63% → 37.02% | 1.40263 → 0.75078 |
| Shifted prior | 0.93215 → 0.78456 | 41.35% → 31.42% | 1.03991 → 0.70783 |

NLL and squared distribution error improve in all three conditions. In conflicting evidence,
however, unweighted expected error increases while cost decreases. The aggregate cost gain
therefore must not be described as fewer mistakes in every condition. These exploratory
subgroups do not establish a causal curriculum advantage before the control comparison.

The original training/development cross-tabulation also shows each uncertainty condition with
all three question types: 106–108 training and 33–34 development questions per combination.
Condition and type are not completely confounded. This is a count check, not evidence of broad
real-world generalization or of independence from pretraining. `epoch-1-conditions.json` preserves
the counts, complete conditional metrics, source identities and scope limitations.

## Second curriculum epoch: score coverage still fails

Both curriculum epochs completed, and the matched outcome-replay control started. The following
are unmerged development results on the same original questions. Neither the merged selection
nor independent calibration/final evaluation has completed.

| Development metric | Parent | Curriculum epoch 1 | Curriculum epoch 2 |
|---|---:|---:|---:|
| New-workflow accuracy | 72.89% | 72.56% | 76.33% |
| Familiar-workflow accuracy | 77.33% | 76.33% | 80.00% |
| Uncertainty NLL | 0.96223 | 0.87228 | 0.87589 |
| Uncertainty error cost at 80% coverage | 1.28726 | 0.84549 | 0.86918 |
| Choice coverage at at most 15% error | 95.89% | 96.32% | 98.80% |
| Score coverage at at most 15% error | 22.19% | 22.34% | 29.69% |
| Noul coverage at at most 15% error | 67.07% | 58.10% | 82.07% |

Noul clears the development coverage screen, while score remains below the required 60%.
At 60% score coverage, expected error is 21.53%, above the 15% maximum. Accuracy and coverage
improved over epoch 1, but uncertainty NLL and cost slightly worsened; every metric did not
improve monotonically. Both remain below the parent's values in this interim
comparison. Photo, NLI and intent development accuracies are 96.67%, 86.89% and 83.98%.
Fact/rule diagnostic hard-label accuracies are 76.70% and 75.00%, using the same unmerged
diagnostic procedure as epoch 1. These auxiliary scores do not replace the original gates.

`skill_curriculum-epoch-2.json` preserves all metrics and model/source identities. All five
candidates' merged comparison remains required. There is no calibration
pass, release approval or demonstrated advantage over matched additional outcome training yet.

## Matched first epoch: prerequisite gains do not establish a decision advantage

The first control epoch also completed. Both columns use unmerged inference, the same original
development questions and the same checkpoint timing: one epoch and 1,882 optimizer updates.
The curriculum's second epoch above must not be compared with this first control epoch to
attribute a benefit to the curriculum.

| Development metric | Curriculum epoch 1 | Outcome-replay epoch 1 |
|---|---:|---:|
| New-workflow accuracy | 72.56% | 73.67% |
| Familiar-workflow accuracy | 76.33% | 74.33% |
| Uncertainty NLL | 0.87228 | 0.85801 |
| Uncertainty error cost at 80% coverage | 0.84549 | 0.74668 |
| Choice coverage at at most 15% error | 96.32% | 97.47% |
| Score coverage at at most 15% error | 22.34% | 18.91% |
| Noul coverage at at most 15% error | 58.10% | 73.28% |
| Fact-query hard-label accuracy | 67.72% | 60.68% |
| Rule-query hard-label accuracy | 68.01% | 42.74% |

The curriculum has higher prerequisite-query and familiar-workflow accuracy, while the control
has higher new-workflow accuracy and lower uncertainty NLL and cost. Neither reaches the required
score coverage; at 60% score coverage, the control's expected error is 23.21%. The control's photo,
NLI and intent development accuracies are 96.33%, 87.11% and 84.20% respectively.

This mixed result does not establish a general decision advantage from the prerequisite
curriculum. These are single-seed, unmerged development point estimates without confidence
intervals. The second matched epoch is reported below; the complete merged comparison remains
pending. Candidate selection and all release gates are unchanged. `outcome_replay-epoch-1.json` preserves
the complete aggregate metrics and checkpoint/source identities. No calibration or final
inference was performed for this interim comparison.

## Matched second epoch: all training finished, score coverage still fails

Both arms completed two epochs and 3,764 optimizer updates. The training coordinator finished
successfully and the five-candidate merged evaluation started. The following comparison still
uses unmerged development inference on the same questions; it is not the selection result.

| Development metric | Curriculum epoch 2 | Outcome-replay epoch 2 |
|---|---:|---:|
| New-workflow accuracy | 76.33% | 75.11% |
| Familiar-workflow accuracy | 80.00% | 79.33% |
| Uncertainty NLL | 0.87589 | 0.86304 |
| Uncertainty error cost at 80% coverage | 0.86918 | 0.89099 |
| Choice coverage at at most 15% error | 98.80% | 98.90% |
| Score coverage at at most 15% error | 29.69% | 24.38% |
| Noul coverage at at most 15% error | 82.07% | 81.72% |
| Fact-query hard-label accuracy | 76.70% | 64.08% |
| Rule-query hard-label accuracy | 75.00% | 46.24% |

The curriculum has higher workflow and prerequisite accuracy and slightly lower uncertainty
cost; the control has lower uncertainty NLL. Both fail the required 60% score coverage. At
60% score coverage, expected error is 21.53%/22.90%, above the 15% limit. The control's photo,
NLI and intent development accuracies are 96.33%, 87.56% and 83.98% respectively.

These are single-seed development point estimates without confidence intervals. Neither the
small workflow differences nor the larger prerequisite-task differences establish independent
final generalization or a release pass. Probability results remain mixed. No candidate is
added and no gate is changed. `outcome_replay-epoch-2.json` preserves the full aggregate metrics
and checkpoint/source identities. Fresh calibration and final inference remain unused while
merged candidate selection is pending.

## Evaluation and package handoff

`compare_curriculum_development.py --wait` prepares the descriptive arm comparison after all
five merged measurements exist, whether selection qualifies or fails. It recomputes development
metrics and coverage from saved raw logits, compares each matched epoch on the same observation
groups, and uses 2,000 paired group-bootstrap replicates with seed 191 for accuracy differences.
Probability and cost differences are reported as point estimates. This comparison is excluded
from candidate selection, introduces no new model, and does not change any release criterion.
Its intervals describe the observed development groups, with no correction for multiple
comparisons; they do not measure training-seed or workflow-family uncertainty.

`cached-development-audit.json` independently reproduces the Foundation and parent development
metrics and risk curves from their saved logits; no GPU inference or calibration/final data was
used. This checks the recorded evidence, not a performance improvement. The same recomputation
will be required for every new candidate before a selected model enters fresh calibration.

`run_curriculum_release_evaluation.py` waits for eligible merged selection and then runs the
remaining gates in order. It records a failure before progressing to later stages. Runtime
versioning and wheel building happen only after training finishes and selection qualifies. The
final package must execute real text/image requests through its bundled wheel in isolated
Python, and its upload helper requires the matching verification report. The current environment
and cached base weights are reused; this does not verify installation on a fresh machine.
None of the future calibration, final or package checks has passed merely because this pipeline
is ready. Actual HF publication still requires a separately configured account and target.

`manifest.json` binds the copied aggregate data audits and pretraining diagnostic reports.
Generated training rows, raw logits and local model/optimizer artifacts are not uploaded here.
