# Fresh independent validation and final populations

[Release/data design](../../../configs/workflow-recovery-release-v13.json) was declared before
these observations were generated. The later [backbone recovery design](../../../configs/workflow-backbone-recovery-v13.json)
reserved the same, then-uninspected populations after every head-only candidate failed.

| Population | Questions | Observation groups | Role |
| --- | ---: | ---: | --- |
| Policy validation, seed 257 | 2,644 | 1,084 | Entire population must pass unchanged per-type error ≤15% and coverage ≥60% |
| Policy validation, seed 263 | 2,644 | 1,084 | Same frozen policy; both populations are mandatory |
| Fresh final, seed 269 | 3,432 | 1,992 | Paired comparison against frozen Foundation after all prerequisites |

Final-only families are equipment reservation, travel reimbursement and supplier onboarding.
Their three Boolean decision truth tables differ from each other and all twelve v12 family
tables. Each family has 160 observation groups with three complete typed views and one uncertain
view. There are 160 uncertain groups per missing/conflicting/shifted-prior condition. The other
1,512 final observations are 600 CIFAR images, 450 SNLI premises and 462 BANKING77 utterances.
Public data and generated cases are not unrestricted real-business deployment evidence.

The independent target auditor reconstructs 2,040 procedural targets in each validation and
1,920 in final directly from rendered rules, including exact conditional probabilities. Public
labels retain pinned source versions. Previous text identities, group IDs, exact image hashes
and perceptual near-duplicates are excluded. CIFAR remains evaluation-only with its upstream
license marked unknown; original records/images are not redistributed here.

## Preserved preparation failures and repairs

1. BANKING77 train had only one unused `contactless_not_working` utterance. The first source
   preparation stopped before writing a selection or generating new records. A
   [declared source-pool repair](../../../configs/workflow-recovery-data-source-repair-v13.json)
   uses the same pinned official upstream test CSV for new validation/final only. Previously
   used normalized utterances remain excluded. Upstream test had at least 29 unused examples
   per category before these selections. No new validation/final record enters training or
   policy fitting; B/C/D remain the explicitly inspected fitting cohorts.
2. Both validations then completed. Final source selection hit an inherited validation-only
   count assertion (304 versus the declared final 912 text observations). The
   [second repair](../../../configs/workflow-recovery-final-source-repair-v13.json) calculates
   the count from the unchanged quotas, verifying existing image bytes before reuse. It
   changes no sample, label, rule, quota, seed or performance gate.

All three populations subsequently passed generation and independent target audits. Original
failures and source versions are preserved. Generation finished before recovery inference.
The selected recovery's [frozen policy](../policy-validation/README.md) passed both independent
validations and the original-group regression. The [new final evaluation](../final-evaluation/README.md)
then began with weights, policy and evaluator sources frozen.
