# Motivation #

In an era of increasing political polarization and widespread disinformation, the ability to verify the truthfulness of online political communication has become both technically and socially vital. Social media platforms, especially Twitter, serve as major channels for political messaging, but also for the rapid spread of fake news. Manual fact-checking is time-consuming and resource-intensive, making automated, AI-driven solutions increasingly relevant.

This project aims to address this challenge by developing a machine learning project that can analyze political tweets and classify them based on their semantic similarity to verified fake or true content. By using modern transformer-based models and multilingual capabilities, we explore a new approach to automated fact assessment.

# Contents #

[1 Introduction ](#1-introduction)

[1.1 Research Question](#11-research-question)

[1.2 Structure of This Document](#12-structure-of-this-document)

[2 Related Work ](#2-related-work)

[2.1 Traditional Approaches to Fake News Detection](#21-traditional-approaches-to-fake-news-detection)

[2.2 Political Tweets and Misinformation](#22-political-tweets-and-misinformation))

[2.3 Multilingual and Cross-Lingual Models](#23-multilingual-and-cross-lingual-models)

[2.4 Gaps and Novelty of Our Approach](#24-gaps-and-novelty-of-our-approach)

[3 Methodology ](#3-methodology)

[3.1 General Methodology ](#31-general-methodology)

[3.2 Data Understanding and Preparation ](#32-data-understanding-and-preparation)

[3.3 Modeling and Evaluation ](#33-model-and-evaluation)

[4 Results ](#4-results)

[4.1 Artifacts](#41-artifacts)

[4.2 Concept of the App](#42-concept-of-the-app)

[5 Discussion ](#5-discussion)

[5.1 Project Outcomes and Contributions](#51-project-outcomes-and-contributions)

[5.2 Limitations](#52-limitations)

[5.3 Ethical Considerations](#53-ethical-considerations)

[5.4 Future Research Directions](#54-future-research-directions)

[6 Conclusion](#55-conclusion)
# 1 Introduction

## 1.1 Research Question

**How can semantic similarity models be used to automatically classify political tweets as true, false, or unclear by comparing them to a curated fake and true news dataset?**

We investigate this question through the implementation of a semantic similarity model based on multilingual sentence embeddings and a user-facing prototype application, assessing both the technical performance and the social implications of such a system.

---

## 1.2 Structure of This Document

This document is structured as follows:

- **Section 2 – Related Work:**  
    Provides an overview of existing research in fake news detection, political NLP, and semantic similarity modelling.
- **Section 3 – Methodology:**  
    Describes the overall research process, including data preparation, model selection, embedding strategy, and classification logic.
- **Section 4 – Results:**  
    Presents the outcomes of our system, both in terms of performance and practical usage scenarios.
- **Section 5 – Discussion:**  
    Discusses limitations, ethical considerations, societal impact, and possible misuse. Also proposes directions for future research.
- **Section 6 – Conclusion:**  
    Summarizes key findings and reflects on the broader relevance of this work.

---

# 2 Related Work

Fake news detection, particularly in political contexts, has become an increasingly important research area due to the widespread influence of misinformation on public opinion and democratic discourse. While most previous approaches have focused on binary classification tasks using large-scale language models, our project introduces a novel multilingual, semantic similarity-based model that performs three-way classification (true, false, uncertain) on political tweets. This represents a new contribution by combining transformer-based sentence embeddings with similarity thresholds in a multilingual political domain.

---

## 2.1 Traditional Approaches to Fake News Detection

Most existing studies classify content as either "fake" or "real." For example, Saeed & Al Solami (2023) explored multiple models—including SVM, CNN, LSTM, and BERT—for fake news detection and achieved accuracy rates between 92% and 99.2%. Alghamdi et al. (2022) compared deep learning and classical methods, reporting F1 scores of up to 93.17% using RoBERTa.

However, these approaches are limited to binary classification and often focus solely on English-language content or domain-specific datasets like PolitiFact and LIAR.

---

## 2.2 Political Tweets and Misinformation

Several studies have applied machine learning to political tweets, but not in the way our model does. Rahim (2021) used a BERT-based approach to detect rumors during the US elections, highlighting challenges with partial truths and the complexity of annotation. Dadkhah et al. (2024) created a massive Twitter dataset (TruthSeeker) with over 180,000 labelled samples for real/fake detection, but again limited the analysis to binary outcomes.

Ornstein et al. (2025) applied large language models like GPT-3 and GPT-4 for political text analysis (sentiment, scaling, topic), but not for veracity classification. Their multi-class classification focuses on content types, not truthfulness.

---

## 2.3 Multilingual and Cross-Lingual Models

Our project stands out through the use of a multilingual semantic similarity model (paraphrase-multilingual-MiniLM-L12-v2) to perform three-way classification (true, false, unclear) of political tweets. In contrast, prior studies such as Kar et al. (2020) and Kazemi et al. (2022) utilized transformer models like mBERT, XLM-RoBERTa, or LaBSE for tasks such as zero-shot learning or multilingual fact-check retrieval. Although Kazemi et al. achieved up to 86% accuracy in cross-lingual fact-check matching, their models were limited to binary outputs and did not address three-class semantic comparison or tweet-level classification.

---

## 2.4 Gaps and Novelty of Our Approach

To date, no known study has used a multilingual semantic similarity model to compare political tweets against a labeled fake news dataset and classify them into three categories: true, false, and unclear. Most existing research:

- Focuses on binary classification (true/false).
- Works with monolingual datasets, primarily in English.
- Does not leverage cross-dataset semantic similarity for classification.

By integrating semantic similarity analysis, multilingual processing, and a three-way classification framework, our project introduces a novel methodological approach for assessing political content and misinformation.

---

# 3 Methodology

## 3.1 General Methodology
Our goal is to provide a lightweight, transparent pipeline that flags potential fake‑news tweets in real time.Firts of all we had to find a way to define fake and true tweets.We used at the beginning TF IDF which is based on text classification.It wasnt precise enough. So we had to substitute our model with Bert(Further explanation about the model in 3.3).With the new model we were able to bind in a reference courpus news_embedding.pkl which has labeled news articles.The next step was to update our Streamlit App.The new Streamlit App has better UI, more functions and performs accuratley with Bert. 

## 3.2 Data Understanding and Preparation
The reference corpus as mentioned before is the multilingual news_embedding.pkl.A large part of the news is german and english.Furthermore the pkl consists of 12176 labeled news items.
For cleaning measurements we lowercased every character, removed URL's and unncessary white space.---Embedding Preparation---


## 3.3 Model and Evaluation
**Model Architecture**\
Tweet → SBERT (512 d) —►  
     │  
     ├─ Mean‐Similarity Score (Fake, True)  
     └─ KNN‑Voting (K = 5)  
               ↓  
        Label ∈ {Fake, True, Unclear}

Each tweet is first embedded with the multilingual SBERT encoder (distiluse-base-multilingual-cased-v1, 512 d); we then compute its mean cosine similarity to the fake and true reference vectors, assigning the label to whichever class yields the higher average (with a small margin threshold to flag “Unclear” cases).

**Training**\
No additional fine‑tuning was required; the SBERT checkpoint is frozen.The reference embeddings were generated offline once with the jupyter Notebook ML4B.ipynb.

**Evaluation**\
On the 20 % test split, the mean-similarity method achieves 83 % accuracy (F1 = 0.78 for Fake, 0.86 for True). The alternative 5-NN voting approach lifts accuracy to 86 % (F1 = 0.81 / 0.85). Both methods are  available in the app, with raw similarity scores displayed so users can gauge the confidence of each prediction.



# 4 Results

## 4.1 Artifacts

* **`App.py`**
  `App.py` functions as both the Streamlit front-end and the model inference backend. It presents a simple web interface where users can either:

  * upload a political tweets dataset and a fake-news reference dataset (CSV/XLSX), or
  * input a single statement via a text box.

  The app detects file format, loads the data with `pandas`, and embeds all texts using the **multilingual transformer model** `paraphrase-multilingual-MiniLM-L12-v2` from the `sentence-transformers` library. Cosine similarity is calculated between each tweet and the fake-news references.

  Classification is based on maximum similarity:

  * **Fake**: score ≥ 0.70
  * **Real**: score ≤ 0.30
  * **Uncertain**: between 0.30 and 0.70

  The result and similarity score (rounded to 3 decimal places) are shown in the app. For Fake classifications, a placeholder panel is rendered to later support LLM-generated justifications.

* **`ML4B.ipynb`**
  The main development and evaluation notebook. It documents:

  * comparisons between TF-IDF and transformer-based embeddings,
  * implementation of **mean similarity** and **maximum similarity** classification strategies,
  * empirical threshold selection,
  * and discussion of evaluation results.
    The notebook also shows code snippets for generating embeddings, visualizing similarity scores, and preparing the app workflow.

    ---

## 4.2 Concept of the App

The app is hosted live at:
🌐 [http://fakenewserkennung.streamlit.app](http://fakenewserkennung.streamlit.app)

It allows users to test political statements in real time using a multilingual semantic similarity approach. The core system relies on:

* `sentence-transformers`: to generate embeddings with `paraphrase-multilingual-MiniLM-L12-v2`.
* `scikit-learn`: to compute cosine similarity and handle TF-IDF in baseline comparisons.
* `pandas`: to process uploaded datasets and export labeled results.
* `nltk` + `stop-words`: to provide a cleaned and filtered German-language pipeline.
* `streamlit`: to render a responsive web interface.

All dependencies are pinned in `requirements.txt`. Development took place in Google Colab (for experimentation) and Visual Studio Code (for modularization and deployment).

### Application Workflow

Step-by-Step Process:
1.Data Upload
Users upload a file (CSV or XLSX) containing political tweets or statements.
Each file must contain a username, created_at, and text column.

2.Semantic Embedding
Each tweet is embedded using the multilingual sentence transformer paraphrase-multilingual-MiniLM-L12-v2.
A pre-cleaned and embedded reference dataset of fake/real news articles is used for comparison.

3.Cosine Similarity Calculation
For each tweet, the app calculates the maximum cosine similarity to the embedded fake-news dataset.

Three-Class Label Assignment
Based on similarity thresholds:

- Fake if similarity ≥ 0.70

- Real if similarity ≤ 0.30

- Uncertain if similarity is between 0.30 and 0.70

4.Result Display
The interface returns:

- The assigned label for each tweet

- Its similarity score (rounded to three decimals)

- And, for Fake labels, an expandable “Explanation” placeholder (for future LLM integration)

Advanced Features:

- See which users have posted the most or least fake news

- Analyze which topics or keywords are most affected by misinformation

- Theme and Usability Options

- Supports Dark Mode for improved readability

- Offers basic sorting and keyword filtering for large datasets
  
---

# 5 Discussion

## 5.1 Project Outcomes and Contributions

Our project produced two core contributions:

1. A semantic similarity-based classification system that labels political tweets as **Fake**, **Real**, or **Uncertain**, using multilingual sentence embeddings from the `paraphrase-multilingual-MiniLM-L12-v2` model.
2. An interactive and publicly accessible [Streamlit application](http://fakenewserkennung.streamlit.app) that enables users to upload political tweet datasets and perform real-time misinformation analysis based on transformer-based embeddings and cosine similarity.

This dual-model approach—transformer-based backend and accessible frontend—balances accuracy with usability. It introduces a **novel three-class classification paradigm** into a research domain that still largely relies on **binary (true/false) decisions**. Furthermore, our system is **language-agnostic**, making it suitable for multilingual misinformation detection across different regions and cultures.

Beyond classification, the app offers **interactive filtering**, such as:

* Ranking users by the volume of misinformation,
* Identifying **topics most affected by fake news**,
* Switching between **Light and Dark Mode** for improved usability.

These features make the tool not only technically robust but also **practically useful** for journalists, researchers, and the general public.

---

## 5.2 Limitations

Despite these contributions, several limitations remain:

* **Data Limitations**:
  Our fake news reference dataset is relatively small and may not cover the full thematic range of political discourse. This can result in **semantic mismatches** and misclassifications.

* **Computational Constraints**:
  All development was conducted on **free Google Colab** and local machines, limiting us from fine-tuning large-scale transformer models or testing on massive multilingual corpora.

* **Simplified Thresholding**:
  Label assignment is based on **fixed cosine similarity thresholds**, which may not generalize across topics or languages. The **“Uncertain”** label, in particular, risks being misinterpreted, as it might simply reflect insufficient reference data rather than genuine ambiguity.

* **Explainability Gaps**:
  Although the app displays similarity scores, it does not yet offer **human-readable justifications**. The placeholder “Explanation” panel in the UI is currently empty, indicating a future extension via LLM integration.

---

## 5.3 Ethical Considerations

Several risks arise from deploying such a system:

* **Misuse in Political Contexts**:
  Automatic labelling of tweets as “Fake” could be misused in **political censorship**, especially when applied without nuance or context.

* **Dataset and Model Biases**:
  Our reference data may reflect biases (e.g., more samples from certain political directions), and **transformer models** themselves may reproduce societal and linguistic biases—especially in **non-English or low-resource languages**.

* **Transparency and Trust**:
  Without clear **rationales for classifications**, users may place undue trust in system outputs or become skeptical of legitimate warnings.

* **Democratic Implications**:
  Algorithmic classification tools must be **complemented by human judgment**, especially in political and journalistic environments. A false sense of certainty can delegitimize alternative perspectives or fuel polarization.

Institutions like [AlgorithmWatch](https://algorithmwatch.org/en/) and the [AI Now Institute](https://ainowinstitute.org/) have emphasized the importance of **auditable, explainable, and accountable** AI systems—principles our project aims to follow in future iterations.

---

## 5.4 Future Research Directions

This project opens up several promising directions for further development:

* **LLM-Driven Explanation Generation**:
  Integrating large language models to generate **natural language explanations** for each classification (e.g., summarizing the similar fake news entry).

* **Dynamic Thresholding**:
  Instead of fixed similarity thresholds, future systems could use **adaptive confidence intervals**, tuned per topic or source.

* **Fact Retrieval Integration**:
  Connecting the system with external **fact-check databases** (e.g., Snopes, PolitiFact) to enhance semantic alignment and factual grounding.

* **Bias-Aware Evaluation**:
  Future work should incorporate **cross-linguistic fairness checks** and **cultural robustness benchmarks** to avoid reinforcing biases in multilingual contexts.

* **Interactive Feedback Loops**:
  Including crowdsourced user feedback can help recalibrate thresholds, detect false positives/negatives, and adapt over time.

**Sample Research Questions**:

* How can semantic similarity systems differentiate between misinformation and manipulative framing?
* What mechanisms ensure multilingual models avoid political bias?
* Can hybrid human-AI systems enhance trust in journalism without suppressing dissent?

---

# 6 Conclusion

This project demonstrates the feasibility and utility of a multilingual, semantic similarity-based system for political misinformation classification. By embedding tweets and comparing them against a labeled fake-news dataset using transformer-based models and cosine similarity, we introduce a scalable and explainable approach to **Fake–Real–Unclear** classification.

Our contributions go beyond academic modeling. The publicly available [Streamlit app](http://fakenewserkennung.streamlit.app) translates research into a **functional prototype** that users can interact with, explore, and analyze politically relevant data in real-time.

Despite limitations related to data scope and interpretability, the system represents a step forward toward **language-aware, user-accessible, and ethically cautious** misinformation detection. Future development will focus on improving explainability, extending multilingual robustness, and integrating external fact sources—moving closer to trustworthy AI support for democratic discourse.

