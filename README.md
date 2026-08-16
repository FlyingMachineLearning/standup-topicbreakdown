# Stand-Up Topic Modeler

A lightweight Streamlit app for exploratory topic modeling of stand-up transcripts.

## What it does

- Upload multiple `.txt` or `.md` transcripts.
- Cleans timestamps and common transcript cues.
- Splits specials/sets into overlapping chunks.
- Uses TF-IDF + Non-negative Matrix Factorization (NMF) to infer topics.
- Shows:
  - topic labels and top terms,
  - corpus-wide topic prevalence,
  - a topic-neighborhood graph,
  - an interactive word co-occurrence web for each topic,
  - representative transcript chunks,
  - transcript-by-topic heatmaps and composition charts,
  - chunk-level topic assignments.
- Exports topics and chunk assignments as CSV.

## Run it

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Input recommendation

One transcript per file works best, for example:

```text
carlin_1996_back_in_town.txt
rock_1996_bring_the_pain.txt
burr_2014_im_sorry_you_feel_that_way.txt
```

The app analyzes **chunks** rather than treating each whole special as a single document. That makes it much more likely to recover actual subject matter instead of merely separating comedians or specials.

## Model choice

This v1 uses TF-IDF + NMF on purpose. It is fast, interpretable, local, and does not need a GPU or model download.

If the corpus gets large or you want semantic topic discovery across different vocabulary ("dating" vs. "relationships" vs. "marriage"), a natural v2 is BERTopic / sentence embeddings with hierarchical topics.
