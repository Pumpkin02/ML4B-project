# Multilingual Fake-News Detection

Similarity-based classification of German political tweets against a labelled English news corpus, using multilingual sentence embeddings. Tweets whose signal is too weak are deliberately labelled **Unclear** instead of being forced into a binary decision.

**Live app:** https://fakenewserkennung.streamlit.app

> **What this is not:** no model is trained or fine-tuned here. The system uses a
> pre-trained sentence-embedding model for inference only and classifies by
> retrieval — nearest neighbours in a fixed, pre-embedded reference corpus.
> `news_embeddings.pkl` stores vectors and labels, not a model.

---

## Approach

**Embedding model:** `distiluse-base-multilingual-cased-v1` (SentenceTransformers), used for inference only. It maps German tweets and English news articles into a shared semantic space, which is what makes the cross-lingual comparison possible at all.

**Reference corpus:** ISOT Fake News Dataset — 44,898 articles (23,481 fake / 21,417 true). Embedded once in advance and cached in `news_embeddings.pkl` as vectors + labels + source texts.

**Classification:** cosine similarity between each tweet embedding and the reference set. Two decision strategies are computed and shown side by side:

| | Mean Similarity | KNN Voting |
|---|---|---|
| Rule | Compare mean similarity to all *fake* vs. all *true* vectors | Take the 5 nearest articles, majority vote on their labels |
| Abstains when | \|sim_fake − sim_true\| < 0.01 | vote margin < 2 (e.g. 3 vs. 2) |
| Strength | Fast, one batch per class | Local neighbourhood, more interpretable |

---

## Results

Evaluated on a held-out split of the **reference corpus**, not on the tweets — the tweets carry no ground-truth labels, so no accuracy can be reported for them.

| Strategy | Coverage | Selective accuracy | Macro-F1 |
|---|---|---|---|
| **KNN Voting** | 81.3 % | **92.6 %** | **0.926** |
| Mean Similarity | 89.6 % | 80.9 % | 0.783 |

*Coverage* = share of items the system actually decided on; *selective accuracy* = accuracy on those items only. Mean Similarity abstains less but is substantially less accurate.

**Why Mean Similarity loses.** In ISOT, every *true* article comes from Reuters, so those texts are stylistically uniform and their embeddings form one tight cluster. Averaging similarity against a tight cluster is systematically depressed relative to averaging against the scattered *fake* sources, which tilts the decision toward *Fake* — fake recall reaches 98.9 % while true recall drops to 55.7 %. KNN Voting looks only at the local neighbourhood and is not affected by this.

---

## Known limitations

These are real and worth stating plainly:

1. **Cross-lingual, cross-domain transfer is unverified.** The reference corpus is long-form English news; the targets are short German tweets. The evaluation above measures performance *within* the reference corpus, so it does not establish that the numbers carry over to tweets.
2. **The model may be reading style, not truth.** All *true* articles are Reuters wire copy, and the source prefixes were not stripped. The classifier may be separating "reads like a Reuters article" from "does not", which is not the same as separating true from false.
3. **Similarity retrieval is not fact-checking.** Nothing here verifies a claim against evidence. A false statement phrased like a news article can score as *True*, and a true statement phrased unusually can score as *Fake*.

The `Unclear` class is a partial mitigation, not a fix: it reduces confident errors but does not make the underlying signal more valid.

---

## Operations

The app was deployed publicly and later stopped starting. Post-mortem:

- **Symptom:** `NameError` on startup; the app no longer booted.
- **Cause:** dependency drift — a `transformers` 5.x release was pulled in that is incompatible with the installed `torch` 2.2.2.
- **Fix:** traced the failure to the version pair, pinned both in `requirements.txt`.
- **Also fixed:** a `KeyError` on `text_clean`, caused by the preprocessing line that creates the column having been commented out; the column is now always created if absent.
- **Added afterwards:** a demo mode, so the app can be tried without uploading any data.

---

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

`news_embeddings.pkl` must be present in the project root.

**Input format:** CSV only. One column named `text` is required (`tweet`, `content`, `full_text` and `text_clean` are accepted as fallbacks). An optional `user` column enables the per-account breakdown. URLs and excess whitespace are stripped automatically.

Or click **Load demo data** in the app to run on a small built-in sample.

---

## Repository

- `app.py` — Streamlit interface and inference
- `news_embeddings.pkl` — cached reference embeddings, labels and texts
- `requirements.txt` — pinned dependencies
- `ML4B.ipynb` — development and evaluation notebook

---

## Context

Built as a project for the *Machine Learning for Business* course at FAU Erlangen-Nürnberg, and maintained since.
