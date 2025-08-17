# BanglaWiC
BanglaWiC: A WiC-Style Polysemy Dataset, Multilingual Transformer Benchmarks, and the Limits of Cross-Lingual Transfer

# Bangla WiC Dataset Construction & WSD Evaluation Pipeline

A complete, end-to-end pipeline to build a Bangla Word-in-Context (WiC) dataset and evaluate word-sense clustering across multiple multilingual sentence-embedding models. It covers: corpus preprocessing, regex tokenization, lemma-aware retrieval with suffix rules, multi-corpus merging, interactive annotation (SahajBERT embeddings + 2D projection), gold/unannotated merging with caps, multi-model clustering & summaries, strict per-lemma feature-level ensemble, and finally WiC dataset construction (capped/uncapped) with zero-shot and few/full-shot evaluation.

Many scripts use Windows paths in the examples; replace with your own paths as needed.

# Table of Contents

- Pipeline Overview
- Repository Structure
- Installation
- Step 1 — Corpus Prep & Retrieval
- Step 2 — Annotation, Merging & Model Evaluation
- Step 3 — Build WiC & Run Evaluations
- Outputs
- Reproducibility
- References (Harvard style)
- License
- Acknowledgements

# Pipeline Overview

1. Preprocess corpora
Filter CSV body fields → regex sentence split (Bangla “।”, !, ?) with short-sentence filter → retrieve sentences for target lemmas using suffix-aware regex → merge & shuffle per lemma across corpora.

2. Annotate & consolidate
Compute lemma offsets for labeled CSVs; create “projected” subsets; interactively annotate sentences via SahajBERT embeddings + UMAP/MDS/t-SNE/PCA; merge annotated + unannotated with caps; evaluate clustering across many models; print/plot session summaries; run a strict per-lemma Top-6 weighted feature-level ensemble.

3. Build WiC & evaluate
Create WSD gold table → generate WiC pairs (capped or uncapped) → zero-shot similarity baselines → few/full-shot fine-tuning with early stopping and consolidated result tables.

Installation
# Python 3.9–3.11 recommended
pip install -r requirements.txt

# If needed individually:
pip install sentence-transformers torch umap-learn scikit-learn pandas numpy matplotlib plotly ipywidgets
# Bangla helpers some scripts use
pip install bangla-stemmer bnlp-toolkit


- For Plotly + ipywidgets interactivity inside Jupyter, ensure widgets are enabled (e.g., JupyterLab 3+ usually works out-of-the-box).
- Install a Bangla-capable font (e.g., Nirmala UI) if your plots show missing glyphs.

# Step 1 — Corpus Prep & Retrieval
Prothom Alo: Batch filter body column
Scans a folder of CSVs, extracts the body column when present, and writes body-only CSVs to an output folder. Creates the folder if missing and logs per-file status.

Regex-based Bangla sentence tokenization (+ length filter)
Reads each body CSV, splits into sentences by Bangla danda “।”, !, ?, removes very short sentences (≤ 2 words), and writes one sentence per line to .txt files.

Keyword-aware sentence retrieval (lemma + suffixes)
From tokenized .txt files, retrieves sentences that contain any target lemma or lemma + common Bangla inflectional suffixes, using a comprehensive boundary-aware regex. Writes per-lemma text files and prints counts.

Multi-corpus merger & shuffler
Given per-lemma .txt sets from multiple corpora (e.g., Prothom Alo, IndicNLP, BNWaC), discovers all lemmas, loads, concatenates, strips blanks, shuffles, and writes one combined file per lemma to a target folder.

# Step 2 — Annotation, Merging & Model Evaluation
Lemma offset annotator (labeled CSVs)
Loads a labeled CSV, infers lemma from the file, finds character start/end offsets of the lemma in each sentence (whole-word preferred; substring fallback), and saves new columns.

Projected dataset creator (first 1,200)
Reads a per-lemma .txt (one sentence per line), keeps the first 1,200 sentences, and writes a UTF-8-SIG CSV.

Interactive annotation with SahajBERT embeddings
Loads up to the first 3000 non-empty sentences, encodes with neuropark/sahajBERT, projects to 2D (UMAP/MDS/t-SNE/PCA), and renders an interactive Plotly scatter where you click to label clusters. Saves labeled selections (lemma, sentence, sense, start, end) as the gold dataset.

Merge annotated + unannotated with caps
Per lemma, normalize whitespace, deduplicate, and enforce a MAX_TOTAL cap that prioritizes annotated rows. Align unannotated to the annotated schema (empty non-sentence fields) and save merged CSVs (UTF-8 with BOM).

Multi-model clustering & performance logging
For each merged lemma CSV:
- Encode sentences with multiple backbones (e.g., MuRIL, XLM-R, IndicBERT, E5, mBERT, BanglaBERT, SahajBERT, LaBSE).
- 2D projection via UMAP for plotting.
- K-Means on annotated only (number of clusters = unique senses).
- Align cluster IDs to gold senses with the Hungarian algorithm, compute Accuracy and macro-F1.
- Plot a grid (lemmas × models), save <lemma>_performance.csv and append to model_performance_master.csv.
- Print session summary table and bar charts (mean ± std).

Strict per-lemma Top-6 weighted feature-level ensemble
Fuse SahajBERT, MuRIL, LaBSE, E5, XL-Lexeme, IndicBERT by:
- L2-normalize → weight by per-lemma F1 (from model_performance_master.csv) → concatenate → optional PCA → UMAP → K-Means (annotated only).
- Save per-lemma CSV and append to an ensemble master; paginate plots over a 6×8 grid.

# Step 3 — Build WiC & Run Evaluations
Build WSD gold table
Sweep a folder of gold files (*_labelled_sentences.csv/.xlsx), validate schema, coerce sense_id, attach optional sense gloss, assign robust sent_id, and write a consolidated CSV (UTF-8 + BOM).

Build WiC (capped/uncapped)
From the gold table, form positive (same-sense) and negative (cross-sense) pairs per lemma:
- Capped: limit partners per sentence (e.g., 32).
- Uncapped: no partner cap.
- Deterministic train/dev/test split via seeded hashing (no sentence overlap across splits), rebalance labels, shuffle, and write wic_train.json, wic_dev.json, wic_test.json + a split summary.

Zero-shot WiC evaluator
Insert [TGT], [/TGT] markers around spans; embed sentence pairs; compute cosine similarity; choose dev threshold that maximizes F1; report test F1/Acc; save predictions and CSV summary for each model.

Few/full-shot fine-tuning
Trainer-free PyTorch loop:
- Add [TGT], [/TGT] to tokenizer, dynamic padding, AdamW + linear warmup/decay.
- Support few-shot regimes (5/10/20/30%) and full-shot (100%).
- Early stopping on dev F1; save the best checkpoint & per-epoch CSV logs.
- Evaluate on test, aggregate to ALL_results.csv and a pivot ALL_results_pivot.csv.

# Outputs
- Per-lemma .txt: extracted & merged sentences.
- Gold CSVs: (lemma, sentence, sense, start, end) from interactive labeling.
- Merged per-lemma CSVs: annotated + unannotated (capped).
- Per-lemma performance CSVs: <lemma>_performance.csv.
- Master performance tables: model_performance_master.csv, ensemble6_wfe_master.csv.
- Plots: grids of UMAP scatter plots (per word × model) and summary bar charts.
- WiC JSONs: wic_train.json, wic_dev.json, wic_test.json + summary.

# Reproducibility
- Random seeds are set where applicable (K-Means, UMAP, train loops).
- Splits are deterministic using seeded hashing of sent_id.
- The strict ensemble requires per-lemma F1 for each of the 6 models; run the per-model evaluator first to populate model_performance_master.csv.

# References (Harvard style)
- Goworek, R. (no date) projecting_sentences. GitHub. Available at: https://github.com/roksanagow/projecting_sentences (Accessed: 18 August 2025).
- Goworek, R., Karlcut, H., Shezad, H., Darshana, N., Mane, A., Bondada, S., Sikka, R., Mammadov, U., Allahverdiyev, R., Purighella, S., Gupta, P., Ndegwa, M., Tran, B.K. and Dubossarsky, H. (no date) ‘SenWiCh: Sense-Annotation of Low-Resource Languages for WiC using Hybrid Methods’. Preprint.
The interactive annotation helper (functions.py) adapts the projecting/labeling workflow from the repository above.

# License
Add your chosen license here (e.g., MIT). If you redistribute data from third-party corpora, ensure you follow their licenses/terms.

# Acknowledgements
- Thanks to the maintainers/authors of Sentence-Transformers, Hugging Face Transformers, UMAP, scikit-learn, and the Bangla NLP community resources.
- Special acknowledgement to the projecting_sentences project and the SenWiCh paper for inspiring the interactive projection + annotation workflow.
