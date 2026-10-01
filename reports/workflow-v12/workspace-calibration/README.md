# Fresh calibration for the separate continuation study

The [design](../../../configs/workflow-workspace-calibration-v12.json) was committed before
candidate selection and declared before generation. These files describe data construction;
they contain no model calibration result. The release remains blocked.

The original public typed-decisions train split has 1,200 cases, all already partitioned in
earlier Veyra work. The latest upstream revision retained the same train Parquet bytes and had
no unused validation split. The author describes the generator as sponsor-accessible. We used
freshly authored explicit policies for the familiar workflow calibration portion and disclosed
the change in label provenance.

| Fresh calibration portion | Questions | Target provenance |
| --- | ---: | --- |
| Nine train/development procedural workflow families | 1,080 | Exact stated rules |
| Missing, conflicting and shifted-prior observations | 360 | Exact conditional evidence model |
| Four familiar workflow domains | 600 | New caller-supplied explicit Boolean policies |
| SNLI | 150 | Original public labels, previously unused premise groups |
| BANKING77 | 154 | Original public labels, previously unused utterances |
| CIFAR-10 | 300 | Original public labels, previously unused images |

Total: **2,644 questions in 1,084 groups**. All original training, development and final records
are preserved after JSONL serialization, including observation identities. No final family
was added to calibration. Exact image and pHash/dHash/color screening covers 5,002 previous
unique images; original dataset bytes are not included in these public reports.

- `protocol.json`: aggregate counts, split hashes, sources and provenance; copied byte-for-byte
  from the generated data protocol.
- `independent-target-audit.json`: independently parses the actual rendered input rules and
  recomputes all **2,040** procedural targets. Maximum absolute difference is **2.22e-16**.
  This checks the declared rules, not real operational event frequencies.

The 600 new familiar-domain questions are **not new teacher annotations from the public
benchmark**. Their calibration results cannot be substituted for results on the original
teacher-labeled population. After fresh fitting is frozen, the unchanged policy must also
achieve error ≤15% and coverage ≥60% **for every type on exactly the old policy-fitting groups**.
Failure stops the sequence before independent final inference. Those old groups remain a
previously inspected regression population; they never rank continuation candidates or refit
the deployed policy.

Sources: [public dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) and
[author's benchmark description](https://latentnode.pages.dev/articles/typed-decisions).
All original release gates remain mandatory. CIFAR-10 is evaluation-only and its upstream
license remains recorded as unknown.
