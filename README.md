# Toxicity Detection & Gender/Race Bias Analysis of Social Media Posts

**Stack:** Python, PyTorch, Hugging Face Transformers (DeBERTa-v3-base), pandas, scikit-learn, scipy, matplotlib

---

## 1. Project Overview

This project trains a transformer-based toxicity classifier on the **Jigsaw Unintended Bias in
Toxicity Classification** dataset, applies it to a large multi-platform **social media dataset
(Bluesky)**, and analyses whether posts that mention gender- or race-related keywords receive
systematically different (model-predicted) toxicity scores than posts that do not.

The workflow is split across **seven Jupyter
notebooks**, each with a single, clearly-scoped responsibility. Every notebook reads the saved
output of the notebook before it — nothing is recomputed silently, and every intermediate result
is written to disk (`.parquet` / `.csv` / `.json` / `.png`) so the whole pipeline is resumable and
auditable.

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_jigsaw_exploration.ipynb` | First look at the Jigsaw training data — target distribution, identity columns, missingness, example comments. |
| 02 | `02_social_media_exploration.ipynb` | First look at the raw Bluesky dataset — platform breakdown, languages, sentiment fields, engagement stats, text length. |
| 03 | `03_preprocessing.ipynb` | Cleans both datasets, builds binary toxicity labels, and produces a balanced 100,000-row train/validation split for model training. |
| 04 | `04-deberta-v3-base-training-main-v1.ipynb` | Fine-tunes `microsoft/deberta-v3-base` for binary toxicity classification and saves the model, tokenizer, and all evaluation artifacts. |
| 05 | `05-gender-race-inference.ipynb` | Validates the gender/race keyword lists against Jigsaw's own identity annotations, then runs the trained model on Jigsaw's validation split. |
| 06 | `06-bluesky-inference.ipynb` | Runs the trained model on the Bluesky posts at scale (chunked + checkpointed), applies the validated keyword lists, and adds a prior-correction to the raw model probabilities. |
| 07 | `07-analysis.ipynb` | Final synthesis notebook — pulls together 04, 05, and 06 into tables, statistical tests, and publication-quality figures. |

---

## 2. Datasets

### 2.1 Jigsaw Unintended Bias in Toxicity Classification (Kaggle)
- Source: `jigsaw-unintended-bias-in-toxicity-classification` Kaggle competition.
- **1,780,822 rows** after cleaning/deduplication, with a continuous `target` toxicity score
  (0–1) and per-comment identity annotations (`male`, `female`, `transgender`, `black`, `white`,
  `asian`, `latino`, etc.).
- A comment is treated as toxic if `target >= 0.5`. At that threshold roughly **8.0%** of
  comments are toxic — a substantially imbalanced dataset.

### 2.2 Social media dataset (Bluesky)
- Source: the Zenodo social-media dataset.
- **15,223,017 posts** in the merged/raw form, spanning many languages (English is only
  **~56%** of the raw merge).
- Columns used: `post_id`, `user_id`, `instance`, `date`, `text`, `sent_label`, `sent_score`.
- This dataset has **no toxicity or identity ground truth** — it is the target of inference, not
  training.

### 2.3 Keyword lists
- `Dataset/gender.txt` — 37 terms (categories, nouns, and bare pronouns such as `he`/`she`/`they`).
- `Dataset/race.txt` — 40 terms (race/ethnicity/nationality words).
- These lists are used to flag Bluesky posts as gender-related / race-related, since Bluesky has
  no identity annotations of its own. Their reliability is explicitly validated in Notebook 05
  before being trusted on Bluesky (see §5).

---

## 3. Environment & How to Run

### 3.1 Requirements
```bash
pip install pandas numpy scikit-learn scipy matplotlib plotly tqdm \
            torch transformers datasets \
            ftfy emoji clean-text nltk pyarrow langdetect
```
Notebook 04 requires a **CUDA GPU** (developed and run on Kaggle's T4 GPU environment); the
other notebooks run comfortably on CPU, though 06 is far faster with a GPU available.

### 3.2 Expected folder layout
```
project/
├── Dataset/
│   ├── train.csv                     # raw Jigsaw training file
│   ├── gender.txt                    # gender keyword list
│   ├── race.txt                      # race keyword list
│   ├── merged/*.jsonl                # raw Bluesky merge (one file per source)
│   └── processed/                    # created by Notebook 03
│       ├── jigsaw_processed.parquet
│       ├── jigsaw_balanced.parquet
│       ├── social_processed.parquet
│       └── train_valid_split/
│           ├── train.parquet
│           └── valid.parquet
├── models/
│   └── deberta_v3_base/
│       ├── best_model/               # created by Notebook 04
│       ├── figures/
│       ├── reports/
│       ├── predictions/
│       │   ├── validation_predictions.csv
│       │   ├── jigsaw_gender_race_validation.parquet
│       │   └── bluesky_shards/
│       └── notebook07_results/
│           ├── figures/
│           ├── tables/
│           └── reports/
└── notebooks/
    ├── 01_jigsaw_exploration.ipynb
    ├── 02_social_media_exploration.ipynb
    ├── 03_preprocessing.ipynb
    ├── 04-deberta-v3-base-training-main-v1.ipynb
    ├── 05-gender-race-inference.ipynb
    ├── 06-bluesky-inference.ipynb
    └── 07-analysis.ipynb
```

Files outside this final workflow are retained under `extras/`, including the unused reference Jigsaw CSV files in `extras/Dataset/`, helper scripts, and the project roadmap PDF.

### 3.3 Run order
1. **`03_preprocessing.ipynb`** — cleans both datasets and writes the balanced 80K/20K
   train/validation split.
2. **`04-deberta-v3-base-training-main-v1.ipynb`** — trains the model (a CUDA GPU is
   recommended) using `Dataset/processed/train_valid_split/` and saves artifacts under
   `models/deberta_v3_base/`.
3. **`05-gender-race-inference.ipynb`** — validates the keyword lists and scores the Jigsaw
   validation set with the trained model and saves the result under
   `models/deberta_v3_base/predictions/`.
4. **`06-bluesky-inference.ipynb`** — scores the Bluesky posts. Start with the default
   `SAMPLE_SIZE = 200_000` to verify the pipeline end-to-end quickly; set `SAMPLE_SIZE = None`
   for the full 15.2M-row run (expect several hours — the notebook checkpoints every 50,000 rows
   so an interrupted run can be resumed rather than restarted).
5. **`07-analysis.ipynb`** — reads the saved outputs of 04/05/06 and produces every table,
   figure, and statistical test described below. **This notebook does not retrain or re-infer
   anything** — it only reads and synthesises.

Notebooks 04–07 are configured with relative local paths. Run them from the `notebooks/`
folder so `Path("..")` resolves to the project root. To run on Kaggle, update the path
configuration cell near the top of each notebook to match the attached Kaggle dataset.

---

## 4. Preprocessing (Notebook 03)

- Missing `comment_text` / `text` rows dropped from both datasets.
- Duplicate comments/posts removed (`drop_duplicates` on the raw text column).
- Text normalisation pipeline (`clean_text`):
  1. `ftfy.fix_text` — repairs mojibake/encoding artefacts.
  2. `emoji.demojize` — converts emoji to text tokens.
  3. URL removal via regex.
  4. `clean-text`'s `clean()` — lower-cases, strips emails/phone numbers/currency symbols.
  5. Whitespace collapsing.
- Binary label: `is_toxic = (target >= 0.5)`.
- **Class balancing:** Jigsaw is read in Arrow row-group batches (to avoid loading 1.78M rows
  into memory at once), split into toxic/non-toxic pools, and **50,000 examples are randomly
  sampled from each class** (seed 42) to build a balanced 100,000-row set.
- **Stratified 80/20 train/validation split** (80,000 / 20,000 rows) is written to
  `Dataset/processed/train_valid_split/`.
- Keyword-based gender/race flagging is deliberately **not** done in this notebook — it happens
  after model inference (Notebooks 05/06) so it can never influence what the model learns.

---

## 5. Model Training (Notebook 04)

| Setting | Value |
|---|---|
| Base model | `microsoft/deberta-v3-base` |
| Task | Binary sequence classification (toxic / non-toxic) |
| Max sequence length | 384 tokens |
| Training examples | 80,000 (40,000 / 40,000 balanced) |
| Validation examples | 20,000 (10,000 / 10,000 balanced) |
| Epochs | 4 (with early stopping, patience = 2, monitored on validation F1) |
| Learning rate | 2e-5 |
| Per-device batch size | 8, gradient-accumulation 2 → effective batch size 16 |
| Precision | FP16 on GPU |
| Optimizer | AdamW |
| Model selection | best checkpoint by validation F1 (`load_best_model_at_end=True`) |
| Seed | 42 |

### Results on the 20,000-row held-out validation set

| Metric | Score |
|---|---|
| Accuracy | 0.8944 |
| Precision | 0.8673 |
| Recall | 0.9314 |
| **F1** | **0.8982** |
| **ROC-AUC** | **0.9611** |

```
              precision    recall  f1-score   support

   Non-Toxic     0.9259    0.8575    0.8904     10000
       Toxic     0.8673    0.9314    0.8982     10000

    accuracy                         0.8944     20000
   macro avg     0.8966    0.8944    0.8943     20000
weighted avg     0.8966    0.8944    0.8943     20000
```

Saved artifacts (all under `models/deberta_v3_base/`):
- `best_model/` — trained weights + tokenizer, ready for inference.
- `figures/` — confusion matrix, ROC curve, precision-recall curve, training-loss/F1 curves.
- `reports/eval_metrics.json`, `classification_report.{txt,json}`, `training_history.csv`,
  `run_summary.json`.
- `predictions/validation_predictions.csv` — per-row prediction probabilities, used later by
  Notebook 05 to cross-check that re-run inference reproduces the original training results.

Recall is intentionally prioritised somewhat over precision by using F1 (not accuracy) as the
model-selection metric — for a toxicity filter, missing a genuinely toxic comment is generally
more costly than a false positive.

---

## 6. Gender/Race Keyword Validation (Notebook 05)

Bluesky has no human-annotated gender/race labels, so the project has to trust the
keyword lists to flag relevant posts. Before trusting them on Bluesky, they are
validated against Jigsaw's own identity annotation columns (`male`, `female`, `black`, `white`,
etc., thresholded at `>= 0.5`), which act as ground truth here.

### Validation result — original lists

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Gender (37 terms) | 0.0941 | 0.9825 | 0.1717 |
| Race (40 terms) | 0.2631 | 0.9222 | 0.4094 |

The gender list's precision is very poor (**0.094**) — driven almost entirely by bare pronouns
(`he`, `him`, `his`, `she`, `her`, `hers`, `they`, `them`, `theirs`), which appear in roughly half
of *all* comments regardless of topic.

### Fix: trimmed gender list

Dropping the nine bare pronouns (37 → 28 terms) and re-validating:

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Gender — original (37 terms) | 0.0941 | 0.9825 | 0.1717 |
| **Gender — trimmed (28 terms)** | **0.4579** | **0.9646** | **0.6210** |
| Race — unchanged (40 terms) | 0.2631 | 0.9222 | 0.4094 |

**Decision rule applied:** switch to the trimmed list only if precision at least doubles *and*
recall stays ≥ 0.75. The trimmed list clears both bars (precision 0.094 → 0.458, recall stays at
0.965) and is adopted as the gender keyword method used everywhere downstream. The race list was
left unchanged — its precision was already reasonable and no comparable single source of noise
was identified. Both the trimmed flag (`gender_related_kw`) and the original-list flag
(`gender_related_kw_full`) are kept in the saved output for transparency.

### Toxicity by identity group (Jigsaw, model-predicted probability)

| Group | Annotation-based mean | Keyword-based mean |
|---|---|---|
| gender_and_race | 0.837 (n=206) | 0.759 (n=418) |
| gender_only | 0.689 (n=1,075) | 0.663 (n=2,061) |
| race_only | 0.848 (n=661) | 0.624 (n=1,506) |
| neither | 0.507 (n=18,045) | 0.500 (n=16,002) |

Output: `models/deberta_v3_base/predictions/jigsaw_gender_race_validation.parquet` (19,987 rows — a small number of rows were lost
because the join back to the processed Jigsaw file is performed on cleaned text, and a handful
of validation rows had no exact text match after cleaning; this gap is called out explicitly in
Notebook 07 rather than silently ignored).

---

## 7. Bluesky Inference (Notebook 06)

- Applies the **finalised** keyword patterns from Notebook 05 (trimmed gender list, unchanged
  race list) directly to Bluesky text — independent of the model.
- Runs the trained DeBERTa model over Bluesky posts in **chunks of 50,000 rows**, writing each
  finished chunk to its own checkpoint shard (`models/deberta_v3_base/predictions/bluesky_shards/shard_XXXXX.parquet`). If the job
  is interrupted, already-completed shards are skipped on re-run rather than being redone.
- Includes a **sample mode** (`SAMPLE_SIZE`, default 200,000 rows) so the entire pipeline can be
  verified end-to-end in well under an hour before committing to the full ~15.2M-row / ~41-hour
  run.
- **Language detection** (`langdetect`) is run alongside inference, because only ~56% of the raw
  Bluesky merge is English, and both the keyword lists and the model are English-only. This lets
  the final analysis report an English-only view as well as an all-languages view.
- **Prior-corrected toxicity probability:** the model was trained on an artificially balanced
  50/50 class split, so its raw output probability systematically overstates how likely a random
  post really is to be toxic relative to Jigsaw's true ~8.0% toxic rate. A closed-form logit-shift
  correction (the standard rare-events / case-control correction, King & Zeng 2001) is applied to
  produce `pred_prob_calibrated` alongside the raw `pred_prob`, without retraining:

```python
def calibrate_prob(p, train_pos_rate, true_pos_rate):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    correction = np.log(
        (true_pos_rate / (1 - true_pos_rate)) / (train_pos_rate / (1 - train_pos_rate))
    )
    adjusted_logit = logit + correction
    return 1 / (1 + np.exp(-adjusted_logit))
```

This correction assumes Jigsaw's toxicity rate is a reasonable population reference for Bluesky;
Bluesky's true toxicity rate is unknown, so this remains a stated assumption rather than a
verified fact.

### Sample-run result (200,000 posts, all languages)

| Group | Raw mean | Calibrated mean | n |
|---|---|---|---|
| gender_and_race | 0.535 | 0.415 | 335 |
| gender_only | 0.243 | 0.186 | 7,439 |
| race_only | 0.229 | 0.171 | 2,303 |
| neither (baseline) | 0.097 | 0.073 | 189,923 |

Restricting to English-only posts shifts every number upward (because non-English "neither" posts
pull the pooled baseline down), which is exactly why both views are reported side by side rather
than only the pooled one.

Output columns: `post_id`, `text`, `gender_related_kw`, `race_related_kw`, `pred_prob`,
`pred_prob_calibrated`, `pred_label`, `detected_lang`, `is_english`, `is_duplicate_text`.

---

## 8. Final Statistical Analysis & Visualisation (Notebook 07)

Notebook 07 is analysis-only: it re-reads the saved outputs of 04, 05, and 06, and does not
retrain the model or rerun inference. Its outputs are written to
`models/deberta_v3_base/notebook07_results/{figures,tables,reports}/`.

### 8.1 Statistical methodology

For every group comparison (e.g. `gender_only` vs. `neither`) the notebook reports:
- Group mean, median, standard deviation, and **95% confidence interval** (Student's t
  interval).
- **Welch's t-test** (does not assume equal variances between groups).
- **Mann–Whitney U test** (non-parametric, robust to the heavy right-skew of toxicity scores).
- **Cohen's d** effect size, so the practical size of a difference can be judged independently of
  sample size / p-value.
- Holm correction applied across the family of comparisons, since several tests are run on the
  same data.

### 8.2 Headline statistical results (20,000-post Bluesky sample)

| Comparison | Mean difference (calibrated) | Welch's t | p (Holm-corrected) | Cohen's d |
|---|---|---|---|---|
| gender_only vs. neither | +0.117 | 9.12 | < 0.001 | 0.53 |
| race_only vs. neither | +0.134 | 5.53 | < 0.001 | 0.63 |
| gender_and_race vs. neither | +0.292 | 4.09 | < 0.001 | — |
| gender_only vs. race_only | −0.017 | −0.63 | 0.53 (n.s.) | — |

Posts flagged as gender-related, race-related, or both consistently score higher on
model-predicted toxicity than posts flagged as neither, and the gap is statistically significant
at every threshold tested. The **gender_only vs. race_only** comparison, however, shows **no
significant difference** — the model does not treat gender-flagged posts as more or less toxic
than race-flagged posts on average.

### 8.3 Explicit interpretation guidance included in the notebook

**Can be claimed:**
- The model's Jigsaw validation performance (F1 0.898, ROC-AUC 0.961).
- The quantified precision/recall of the keyword lists against Jigsaw's identity
  annotations.
- That the trimmed gender list is methodologically preferable to the original list.
- That Bluesky posts flagged by keyword as gender/race-related receive higher **model-predicted**
  toxicity scores than posts that are not flagged.

**Should not be claimed:**
- That Bluesky model scores equal validated, human-checked toxicity accuracy (there is no human
  ground truth for Bluesky in this pipeline).
- That keyword-based groups represent users' actual gender or race identity.
- Causal claims — a higher predicted-toxicity score in a keyword group is a correlational
  finding, not evidence of discrimination or model bias by itself.
- That the prior-corrected probability is a verified ground-truth probability for Bluesky.
- That statistical significance (very achievable at this sample size) is the same as a large or
  important effect — the Cohen's d values are reported specifically so effect size, not just
  p-value, drives interpretation.

### 8.4 Figures produced (300 DPI PNGs)

1. DeBERTa validation metrics bar chart
2. Training loss / validation F1 curves
3. Jigsaw confusion matrix
4. Jigsaw ROC curve
5. Jigsaw precision-recall curve
6. Gender/race keyword validation (precision/recall/F1 bar chart)
7. Jigsaw toxicity by identity group
8. Bluesky raw toxicity distribution (histogram)
9. Bluesky prior-corrected toxicity distribution (histogram)
10. Bluesky raw toxicity by keyword group
11. Bluesky prior-corrected toxicity by keyword group

### 8.5 Tables produced (CSV)

Model metrics, recomputed validation metrics, keyword validation metrics, Jigsaw toxicity-by-group,
Bluesky toxicity-by-group with 95% CIs, the full statistical test table, and a compact
"project dashboard" table collecting every headline number in one place for easy citation.

A fully auto-generated Markdown findings report (`notebook07_findings.md`) is also written
directly from the computed tables — this means the exact numbers always match what's in the CSVs,
even if the notebook is re-run against a different sample size later.

---

## 9. Summary of Deliverables

- 7 Jupyter notebooks (01 → 07) with inline markdown explanations.
- Trained `DeBERTa-v3-base` toxicity model + tokenizer (`best_model/`).
- Processed/balanced datasets and train/validation split (Parquet).
- Jigsaw validation-set inference with gender/race grouping (`jigsaw_gender_race_validation.parquet`).
- Bluesky toxicity + bias inference output (`bluesky_toxicity_bias_inference_*.parquet`).
- Full statistical analysis (CIs, Welch's t-test, Mann–Whitney U, Cohen's d, Holm correction).
- 11 publication-quality figures + supporting CSV tables.
- Auto-generated Markdown findings summary.
- This documentation file.

## 10. Known Limitations

- Bluesky toxicity scores are **model inferences**, not human-labelled ground truth — there is no
  way to measure the model's real-world accuracy on Bluesky directly.
- Keyword-based gender/race flagging is a proxy for topical relevance, not for the identity of
  the poster, and inherits whatever recall/precision trade-offs are documented in §6.
- Non-English posts (~44–49% of Bluesky, depending on sample) are scored by an English-trained
  model and matched against English keyword lists; results are reported both pooled and
  English-only for this reason.
- The prior-correction in §7 assumes Jigsaw's toxicity prevalence is representative of Bluesky's
  true toxicity prevalence, which is not independently verified.
- Large sample sizes make statistical significance easy to achieve; effect sizes (Cohen's d) are
  reported specifically so this doesn't get mistaken for practical importance.
