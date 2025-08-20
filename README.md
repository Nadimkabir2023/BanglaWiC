# BanglaWiC

*A WiC-Style Polysemy Dataset and Multilingual Transformer Benchmarks*

> **Author:** Nadim Kabir • **Supervisor:** Dr Haim Dubossarsky
> **Course:** Big Data Processing (MSc Big Data Science)

---

## TL;DR

This repo releases **Bangla-WiC**, a span-aware Bangla Word-in-Context dataset and code to:

1. Harvest & prepare corpora,
2. Annotate via projection-assisted labeling
3. Build **capped/uncapped** WiC splits,
4. Run **zero-shot**, **few-/full-shot fine-tuning**, and a **weighted feature-level ensemble (WFE)**.

**Headline results:** 
- Best single encoder **SahajBERT** (Acc **0.842** / F1 **0.837**)
- **WFE** (+0.040 Acc / +0.041 F1).
- Fine-tuned WiC on the uncapped set peaks at **F1 0.891**.

---

## What’s inside

```
PROJECTING_SENTENCES-MAIN/
├─ Dataset/
│  ├─ bangla_wic_dataset_capped/
│  │  ├─ wic_train.json • wic_dev.json • wic_test.json • summary.json
│  ├─ bangla_wic_dataset_uncapped/
│  │  ├─ wic_train.json • wic_dev.json • wic_test.json • summary.json
│  ├─ Gold_dataset/               # gold, sense-labeled sentences (+ spans)
│  ├─ Merged_dataset/             # per-lemma annotated+unannotated merges
│  └─ bangla_wsd_dataset.csv      # consolidated gold WSD table
├─ Result/
│  ├─ wsd_perf/                   # per-model clustering perf (ours/classmate)
│  ├─ wsd_perf_ensemble/          # strict per-lemma ensemble sweeps
│  ├─ zero_shot_result_{capped,uncapped}/
│  └─ WIC_finetuned_result_{capped,uncapped}/
├─ functions.py                   # interactive projection + span utilities
├─ s1_corpus_prep.ipynb           # Step 1: corpus prep & retrieval
├─ s2_annotation_merge_eval_ensemble.ipynb  # Step 2: annotate + merge + eval + ensemble
└─ s3_wic_build_train.ipynb       # Step 3: build WiC + zero-shot + fine-tune
└─ Bengla_sense_lists.ods         # Bengla sense lists
```

### Short file names (suggested aliases)

* **Step 1:** `s1_corpus_prep.ipynb` (alias: `step1_corpus.ipynb`)
* **Step 2:** `s2_annotation_merge_eval_ensemble.ipynb` (alias: `step2_annot_eval.ipynb`)
* **Step 3:** `s3_wic_build_train.ipynb` (alias: `step3_wic_train.ipynb`)

---

## Setup

**Python** ≥ 3.9. Recommended to use a fresh venv/conda env.

```bash
pip install -U pandas numpy scikit-learn umap-learn plotly ipywidgets
pip install -U torch torchvision torchaudio  # pick CUDA/CPU build as appropriate
pip install -U transformers sentence-transformers
```

---

## Data sources (high level)

* **IndicNLP Bangla (tokenised)** – curated monolingual text (no extra sent-tokenisation).
* **BNWaC via SketchEngine (tokenised)** – broad web text with de-duplication.
* **Prothom Alo news archive (CSV)** – modern news; we extract the **`body`** field and sentence-segment.

All released files here are **sentence-level** and free of personal identifiers. Please respect the original licences of the upstream corpora you harvest.

---

## Reproduce the pipeline

### Step 1 — Corpus prep & retrieval (`s1_corpus_prep.ipynb`)

* Extract `body` from Prothom Alo CSVs → sentence-tokenise (Bangla regex), drop ultra-short lines.
* Keyword-aware retrieval with suffix matching (per-lemma sentence pools).
* Merge IndicNLP / BNWaC / Prothom Alo per lemma and **shuffle** to reduce corpus-order bias.

### Step 2 — Annotation, merge, diagnostics, ensemble (`s2_annotation_merge_eval_ensemble.ipynb`)

* **Interactive projection & labeling**: encode sentences (e.g., SahajBERT), reduce (UMAP/t-SNE), click-label clusters, save **gold** CSV with offsets.

  * Utilities include robust target span finding: exact whole-word first, then Levenshtein fallback (see `get_positions(...)` in **`functions.py`**) .
* Build **merged** per-lemma CSVs (annotated first, capped total).
* **Unsupervised diagnostics**: K-Means on the annotated subset + Hungarian alignment → **Acc / macro-F1** per lemma & model.
* **WFE**: strict per-lemma fusion of the **top-6** backbones (SahajBERT, MuRIL, LaBSE, E5, XL-lexeme, IndicBERT) weighted by per-lemma F1.

### Step 3 — Build WiC & train/evaluate (`s3_wic_build_train.ipynb`)

* Convert the gold WSD table into **WiC**: positives (same sense), negatives (cross sense), **sentence-disjoint** splits.
* Two pairing **modes**: **capped** (MAX\_PARTNERS\_PER\_SENT=32) and **uncapped** (no cap).
* **Zero-shot**: encode pairs with \[TGT]…\[/TGT], sweep a dev threshold, report test **Acc/F1**.
* **Few-/full-shot fine-tuning**: 5%→100% regimes with early stopping on dev macro-F1; save best checkpoints & CSV summaries.

---

## Key results

* **Unsupervised (annotated subset, K-Means + Hungarian):**

  * Best single model: **SahajBERT** — Acc **0.842**, F1 **0.837**.
  * **WFE (top-6)** — Acc **0.882**, F1 **0.878** (**+0.040 / +0.041** over SahajBERT).
* **Zero-shot WiC:** modest; ranking depends on pairing density (capped vs uncapped).
* **Fine-tuned WiC (uncapped):** up to **F1 0.891** (SahajBERT), LaBSE/MuRIL close behind.

Outputs are written under `Result/`:

* `wsd_perf/`, `wsd_perf_ensemble/` – clustering diagnostics (CSVs + plots).
* `zero_shot_result_{capped,uncapped}/` – per-model zero-shot summaries/predictions.
* `WIC_finetuned_result_{capped,uncapped}/` – regime curves, checkpoints, pivots.

---

## Models you can plug in

mBERT, XLM-R, **MuRIL**, IndicBERT, **SahajBERT**, **LaBSE**, E5, **XL-lexeme**, BanglaBERT (and paraphrase SBERTs).
Add/change backbones in the model lists inside the notebooks.

---

## License

The code and dataset are shared for academic, non-commercial use. Respect upstream corpus licences.

---

## Acknowledgments

Thanks to **Prof. Haim Dubossarsky** for guidance, **Swarnendu Moitra** for assistance, and the **QMUL MSc BDS** programme. Community models & datasets credited throughout the notebooks and references.

