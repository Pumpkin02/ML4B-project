
# 1 Introduction

## 1.1 Motivation
In an era of increasing political polarization and widespread disinformation, verifying the truthfulness of online political communication has become both technically and socially vital. Social media platforms—especially Twitter—serve as major channels for political messaging but also for the rapid spread of fake news. Manual fact‑checking is time‑consuming and resource‑intensive, making automated, AI‑driven solutions increasingly relevant.

This project addresses the challenge by developing a machine‑learning workflow that analyses political tweets and classifies them based on their semantic similarity to verified *fake* or *true* content. Leveraging modern transformer‑based models and multilingual capabilities, we explore a new approach to automated fact assessment.

## 1.2 Research Question
**How can semantic‑similarity models automatically classify political tweets as *true*, *false*, or *uncertain* by comparing them with a curated fake‑news corpus?**

We investigate this through a paraphrase‑based semantic‑similarity model and a user‑facing prototype, assessing both technical performance and social implications.

## 1.3 Structure of This Document
* **Section&nbsp;2 – Related Work**  
  Overview of existing research in fake‑news detection, political NLP, and semantic‑similarity modelling.  
* **Section&nbsp;3 – Methodology**  
  Details of data preparation, model selection, embedding strategy, and classification logic.  
* **Section&nbsp;4 – Results**  
  Outcomes of our system in terms of performance and practical usage scenarios.  
* **Section&nbsp;5 – Discussion**  
  Limitations, ethical considerations, societal impact, and directions for future research.
* **Section&nbsp;6 – Conclusion**  
  Key findings and broader relevance of this work.

---

# 2 Related Work

Fake‑news detection—particularly in political contexts—has become a critical area of research due to the growing impact of false information on public discourse. Most previous work has focused on **binary classification** and large‑scale language models, but none has directly implemented a **paraphrase‑based, three‑way classification** of political tweets using multilingual techniques.

## 2.1 Traditional Approaches to Fake News Detection
Most studies label content as either *fake* or *real*. For example, Saeed & Al Solami (2023) explored SVM, CNN, LSTM, and BERT, achieving 92–99 % accuracy. Alghamdi et al. (2022) compared deep‑learning and classical methods, reporting F1 scores up to 93.17 % with RoBERTa. However, these studies focus mainly on English and domain‑specific datasets like *PolitiFact* and *LIAR*.

## 2.2 Political Tweets and Misinformation
Rahim (2021) used a BERT‑based approach to detect rumours during US elections, underscoring challenges with partial truths. Dadkhah et al. (2024) built **TruthSeeker**, a 180 k‑sample Twitter corpus, but again limited analysis to binary outcomes. Ornstein et al. (2025) applied GPT‑3/4 for political text analysis (sentiment, scaling, topic) but not veracity.

## 2.3 Multilingual and Cross‑Lingual Models
Our project leverages a multilingual paraphrase model (*paraphrase‑multilingual‑MiniLM‑L12‑v2*). Prior cross‑lingual studies (Kar 2020; Kazemi 2022) used mBERT, XLM‑RoBERTa, LaBSE for zero‑shot fact‑checking, with Kazemi achieving 86 % accuracy across languages—yet still binary.

## 2.4 Gaps and Novelty of Our Approach
No prior work combines:

* Semantic‑similarity embeddings  
* Multilingual capability  
* **Three‑tier classification (true / false / uncertain)**

Our system fills this gap.

---

# 3 Methodology

## 3.1 General Methodology
We built a two‑part system:

1. **Backend pipeline** for cleaning and preparing large‑scale tweet data.  
2. **Frontend Streamlit app** for manual testing via text‑similarity detection (TF‑IDF + cosine).

**Workflow**

1. **Data Collection**  
   * Bundestag tweets (`.jl`)  
   * Fake‑news dataset (labelled *true* / *false*)
2. **Automated Pre‑processing** (`datacleaning.py`)  
   * Remove emojis, mentions, hashtags, URLs  
   * Preserve German umlauts (ä, ö, ü) and ß  
   * Lower‑case text
3. **Semantic Embedding & Similarity**  
   * Embed tweets and fake/true news with *MiniLM‑L12‑v2*  
   * Compute cosine similarity; label as **True**, **False**, **Unsure**
4. **Threshold‑based 3‑way Classification**  
   * `similarity ≥ τ_true` ⇒ *True*, `≤ τ_false` ⇒ *False*, otherwise *Unsure*
5. **Streamlit App** (TF‑IDF Checker)  
   * Upload datasets, enter statement, receive label (+ score)
6. **Evaluation**  
   * Manual review of a tweet subset and app testing with known claims.

## 3.2 Data Understanding and Preparation
**Datasets**

| Corpus | Format | Key Fields |
|--------|--------|-----------|
| Bundestag Twitter 2022 | `.jl` | `text`, `created_at`, `username`, … |
| Real & Fake News (Kaggle) | `.csv` | `title`, `text`, `subject`, `date` |

Challenge: divergent formats and data richness between corpora.

## 3.3 Modeling and Evaluation
Bag‑of‑words TF‑IDF baseline:

* Clean text (lower‑case, remove URLs/hashtags/mentions/emojis, strip German stop‑words).  
* Train `TfidfVectorizer` (5 000 features) on combined corpus.  
* Cosine similarity to fake‑news vectors.  
* Thresholds: `≥ 0.70 → Fake`, `≤ 0.30 → Real`, else **Uncertain**.  
* Evaluate on a labelled validation split; sweep thresholds for balance.

---

# 4 Results

## 4.1 Artifacts
* **`app.py`** – Streamlit front‑end + inference engine  
* **`datacleaning.py`** – tweet‑cleaning utility  
* **`ML4B.ipynb`** – exploratory notebooks

## 4.2 Concept of the App
* **Stack**: Python 3.10, `scikit‑learn 1.7.0`, `pandas 2.2.3`, `nltk 3.9.1`, `streamlit 1.45.1`  
* **Workflow**: User uploads fake‑news reference + tweet batch *(CSV/XLSX)* → TF‑IDF vectorise → cosine similarity → label + score.

---

# 5 Discussion

## 5.1 Project Outcomes and Contributions
1. **Semantic‑similarity classifier** (true / false / unsure) using multilingual embeddings.  
2. **Interactive Streamlit app** for on‑the‑fly fact checking.

Introduces three‑way classification to a dominantly binary field; multilingual by design.

## 5.2 Limitations
* **Data coverage** – topic/style gaps impair matching.  
* **Compute** – free Colab constraints limit fine‑tuning.  
* **App simplification** – TF‑IDF lacks deep semantics and explanations.  
* **“Unsure” ambiguity** – fixed thresholds may mislead.

## 5.3 Ethical Considerations
* **Misuse** – oversimplified labels may fuel censorship.  
* **Bias** – dataset skews can amplify political bias; cross‑lingual performance varies.  
* **Transparency** – similarity scores lack rationale; affects trust.  
* **Social impact** – algorithmic fact‑checking must keep human oversight.  

See [AlgorithmWatch](https://algorithmwatch.org/en/) and [AI Now Institute](https://ainowinstitute.org/) for broader critiques.

## 5.4 Future Research Directions
* Retrieval‑based fact alignment (Snopes, PolitiFact).  
* Context‑aware classification (timelines, networks).  
* Counterfactual explanations.  
* Crowdsourced evaluation.  

Example research questions:

* Can transformers detect *manipulative framing* rather than outright falsehoods?  
* How to mitigate cross‑lingual political bias in semantic‑similarity models?  
* What role should automation play in journalistic accountability?

---

<!-- optional -->
# 6 Conclusion
*(to be written)*

---

## Table of Contents
<details>
<summary>Open / Close</summary>

- [1 Introduction](#1-introduction)
  - [1.1 Motivation](#11-motivation)
  - [1.2 Research Question](#12-research-question)
  - [1.3 Structure of This Document](#13-structure-of-this-document)
- [2 Related Work](#2-related-work)
  - [2.1 Traditional Approaches to Fake News Detection](#21-traditional-approaches-to-fake-news-detection)
  - [2.2 Political Tweets and Misinformation](#22-political-tweets-and-misinformation)
  - [2.3 Multilingual and Cross-Lingual Models](#23-multilingual-and-cross-lingual-models)
  - [2.4 Gaps and Novelty of Our Approach](#24-gaps-and-novelty-of-our-approach)
- [3 Methodology](#3-methodology)
  - [3.1 General Methodology](#31-general-methodology)
  - [3.2 Data Understanding and Preparation](#32-data-understanding-and-preparation)
  - [3.3 Modeling and Evaluation](#33-modeling-and-evaluation)
- [4 Results](#4-results)
  - [4.1 Artifacts](#41-artifacts)
  - [4.2 Concept of the App](#42-concept-of-the-app)
- [5 Discussion](#5-discussion)
  - [5.1 Project Outcomes and Contributions](#51-project-outcomes-and-contributions)
  - [5.2 Limitations](#52-limitations)
  - [5.3 Ethical Considerations](#53-ethical-considerations)
  - [5.4 Future Research Directions](#54-future-research-directions)
- [6 Conclusion](#6-conclusion)

</details>
