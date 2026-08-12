# How to Run This Project

This guide covers two ways to run the pipeline:

1. **Locally** (Notebooks 01–03, and 04–07 if you have a decent GPU) — set up a `Dataset/`
   folder next to the notebooks and run in order.
2. **On Kaggle** (recommended for Notebooks 04–06) — upload the data as a Kaggle Dataset and run
   the notebooks in a Kaggle Notebook session with a GPU turned on. Notebook 04 (training) and
   Notebook 06 (full 15.2M-row inference) are heavy enough that this is the realistic option
   unless you have a local CUDA GPU with a good amount of VRAM and are prepared to wait hours.

Notebooks 04, 05, 06, and 07 use relative local paths and expect to be run from inside the
`notebooks/` folder. To run them on Kaggle, update the single path-configuration cell near the
top of each notebook to point at the attached dataset and a writable output directory.

---

## 1. Local setup (Notebooks 01–03)

### 1.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.2 Folder layout to create

Notebooks 01–03 use **relative paths** (`Path("..")`, `Path("../Dataset/...")`), so they expect
to be run from inside a `notebooks/` folder that sits next to a `Dataset/` folder. Build this
structure and put the raw files in it before running anything:

```
project/
├── Dataset/
│   ├── train.csv                # raw Jigsaw training file (from the Kaggle competition)
│   ├── gender.txt               # gender keyword list (37 terms)
│   ├── race.txt                 # race keyword list (40 terms)
│   ├── merged/                  # raw Bluesky merge — one .jsonl file per source
│   │   ├── source1.jsonl
│   │   ├── source2.jsonl
│   │   └── ...
│   └── processed/                # created automatically by Notebook 03 — leave empty/absent
├── models/
│   └── deberta_v3_base/
│       ├── best_model/
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

Where to get the raw data:
- `Dataset/train.csv` — download from the Kaggle competition
  **"Jigsaw Unintended Bias in Toxicity Classification."**
- `Dataset/merged/*.jsonl` — the Zenodo Bluesky social-media dataset.
- `Dataset/gender.txt` / `Dataset/race.txt` — the keyword lists.

Files outside this final workflow remain under `extras/`, including the unused reference Jigsaw CSV files in `extras/Dataset/`, helper scripts, and the project roadmap PDF.

### 1.3 Run order locally

1. **`01_jigsaw_exploration.ipynb`** and **`02_social_media_exploration.ipynb`** — read-only
   exploration, no paths need editing, just run top to bottom once §1.2 is in place.
2. **`03_preprocessing.ipynb`** — reads `Dataset/train.csv` and `Dataset/merged/`, writes to
   `Dataset/processed/` (created automatically). Produces:
   - `Dataset/processed/jigsaw_processed.parquet`
   - `Dataset/processed/jigsaw_balanced.parquet`
   - `Dataset/processed/social_processed.parquet`
   - `Dataset/processed/train_valid_split/train.parquet`
   - `Dataset/processed/train_valid_split/valid.parquet`
3. **Notebooks 04–07** — already configured for this local structure; run them from inside
   `notebooks/`.

---

## 2. Local path configuration

Notebooks 04–07 use relative paths for the local structure above. Run them from inside the
`notebooks/` folder, where `Path("..")` resolves to `project/`.

| Notebook | Local paths |
|---|---|
| 04 | Inputs: `Dataset/processed/train_valid_split/`; outputs: `models/deberta_v3_base/` |
| 05 | Processed files: `Dataset/processed/`; keywords: `Dataset/gender.txt` and `Dataset/race.txt`; output: `models/deberta_v3_base/predictions/jigsaw_gender_race_validation.parquet` |
| 06 | Processed Bluesky data: `Dataset/processed/social_processed.parquet`; model: `models/deberta_v3_base/best_model/`; outputs: `models/deberta_v3_base/predictions/` and `predictions/bluesky_shards/` |
| 07 | Inputs: `models/deberta_v3_base/` and its `predictions/` directory; results: `models/deberta_v3_base/notebook07_results/` |

To run on Kaggle instead, update the same configuration cells to point at your attached Kaggle
dataset and a writable Kaggle output directory.

---

## 3. Hardware/time limitations — why Kaggle is the realistic option for 04 and 06

Two steps in this pipeline are genuinely heavy, and running them on an underpowered local
machine (no GPU, or a small laptop GPU) either fails outright (out-of-memory) or takes far too
long to be practical:

- **Notebook 04 (model training):** fine-tunes `microsoft/deberta-v3-base` on 80,000 examples
  for 4 epochs at 384-token sequence length. **Requires a CUDA GPU.** On a Kaggle T4, this took
  roughly **1 hour 52 minutes** (`train_runtime` ≈ 6,732s). On CPU this is not practical — it
  would likely take days.
- **Notebook 06 (Bluesky inference at scale):** the full dataset is **15,223,017 rows**. Even on
  a Kaggle T4 GPU this is a **multi-hour job (~29 hours** for the full run at the batch size
  used). This is why the notebook defaults to `SAMPLE_SIZE = 200_000` — it lets you verify the
  entire pipeline end-to-end in well under an hour before committing to the full run. The
  notebook checkpoints every 50,000 rows to its shard directory, so an interrupted run resumes
  instead of restarting from scratch.

If you hit "CUDA out of memory," an install/runtime error caused by no GPU being available, or a
run that's taking many hours locally, **stop and move the run to Kaggle** using the steps below
rather than trying to force it through on local hardware.

---

## 4. Running on Kaggle with a GPU (recommended for Notebooks 04 and 06)

### 4.1 Package the data as a Kaggle Dataset

1. On your machine, gather everything the notebooks need into one folder, matching the structure
   the local structure in §2:
   - `Dataset/processed/train_valid_split/train.parquet` and `valid.parquet` (from local
     Notebook 03, or produce them first)
   - `Dataset/processed/jigsaw_processed.parquet`, `jigsaw_balanced.parquet`,
     `social_processed.parquet`
   - `Dataset/gender.txt`, `Dataset/race.txt`
   - once you have it, `models/deberta_v3_base/best_model/` (the trained model from
     Notebook 04 — needed as an input for Notebooks 05 and 06)
2. Go to **kaggle.com → Datasets → New Dataset**.
3. Give it a title. Update the configuration paths in the notebooks to match the slug and Kaggle
   output directory you use.
4. Drag and drop the folder(s) from step 1, or use the Kaggle API:
   ```bash
   pip install kaggle
   kaggle datasets create -p /path/to/your/folder -r zip
   ```
5. Wait for the upload/processing to finish, then note the dataset's path — it will be mounted
   at `/kaggle/input/<your-dataset-slug>/` inside any notebook you attach it to.

### 4.2 Create the Kaggle Notebook and turn on the GPU

1. Go to **kaggle.com → Code → New Notebook**.
2. Click **File → Upload Notebook** and upload the `.ipynb` you want to run (e.g.
   `04-deberta-v3-base-training-main-v1.ipynb`).
3. In the right-hand sidebar, click **Add Input → Datasets**, search for the dataset you created
   in §4.1, and attach it. It will appear under `/kaggle/input/`.
4. Turn on the GPU:
   - In the right-hand sidebar, open the **Session options** panel (sometimes labelled
     **Accelerator**).
   - Change the accelerator dropdown from **None** to a GPU option. Kaggle currently offers
     **GPU T4 x2** and, where available on your account, **GPU P100** — pick whichever the
     newest/fastest option available to you is (Notebook 04's comments assume a T4, and
     explicitly pin `CUDA_VISIBLE_DEVICES=0` to avoid a DataParallel/fp16 bug when two GPUs are
     visible, so a single-GPU or dual-T4 accelerator both work without further changes).
   - Kaggle GPU sessions have a weekly quota (historically ~30 GPU-hours/week, subject to
     change) and a per-session time limit — check the current limits under **Settings → Account**
     if a long run risks running out of quota mid-way.
5. Turn on internet access if it isn't already on (**Session options → Internet → On**) — needed
   to download the `microsoft/deberta-v3-base` weights from Hugging Face and to `pip install`
   packages.
6. In the first cell, install anything not already present in the Kaggle image:
   ```python
   !pip install -q clean-text ftfy emoji langdetect
   ```
7. Verify the GPU is actually visible before running the rest of the notebook:
   ```python
   import torch
   print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
   ```
8. Update `DATA_DIR` / `ROOT` / `MODEL_DIR` and the output path in the config cell to your
   attached dataset and writable Kaggle output location, then run the notebook top to bottom
   (**Run → Run All**, or cell by cell).

### 4.3 Getting outputs back out and chaining notebooks together

Kaggle notebook outputs are written to `/kaggle/working/`, which is only saved when you commit
the notebook (**Save Version → Save & Run All**). After a version finishes:

1. Open the saved version's **Output** tab and download the files you need (e.g.
   `best_model/`, `jigsaw_gender_race_validation.parquet`,
   `bluesky_toxicity_bias_inference_*.parquet`).
2. To feed one notebook's output into the next (e.g. Notebook 04's `best_model/` into
   Notebook 05), either:
   - Add the first notebook's saved output directly as a data source (**Add Input → Notebook
     Output**) on the next notebook, or
   - Download it and re-upload it into the same Kaggle Dataset from §4.1 (creating a **new
     version** of that dataset) so `ROOT` continues to resolve everything from one place, which
     is how Notebooks 05/06/07 are already structured to expect it.

### 4.4 Recommended order on Kaggle

1. **Notebook 04** on GPU — train the model (~2 hours on a T4). Save the version, download/attach
   `best_model/`.
2. **Notebook 05** on GPU (or CPU — it's just inference on 20K rows) — validates the keyword
   lists and produces `jigsaw_gender_race_validation.parquet`.
3. **Notebook 06** on GPU — start with the default `SAMPLE_SIZE = 200_000` to confirm everything
   works end-to-end in under an hour. Only set `SAMPLE_SIZE = None` for the full 15.2M-row /
   ~29-hour run once you're confident and have the GPU quota to spare — consider splitting it
   across multiple sessions using the checkpoint/resume behavior if you're limited by Kaggle's
   per-session time cap.
4. **Notebook 07** — analysis-only, no GPU required, reads the saved outputs of 04/05/06 and
   produces the final tables/figures/report. Fine to run on CPU, even locally, once you've
   downloaded the outputs of 04/05/06 from Kaggle.
