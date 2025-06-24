# 1 Introduction

**1\. Introduction**

**1.1 Motivation**

In an era of increasing political polarization and widespread disinformation, the ability to verify the truthfulness of online political communication has become both technically and socially vital. Social media platforms, especially Twitter, serve as major channels for political messaging, but also for the rapid spread of fake news. Manual fact-checking is time-consuming and resource-intensive, making automated, AI-driven solutions increasingly relevant.

This project aims to address this challenge by developing a machine learning project that can analyze political tweets and classify them based on their semantic similarity to verified fake or true content. By using modern transformer-based models and multilingual capabilities, we explore a new approach to automated fact assessment.

**1.2 Research Question**

**How can semantic similarity models be used to automatically classify political tweets as true, false, or uncertain by comparing them to a curated fake news dataset?**

We investigate this question through the implementation of a paraphrase-based semantic similarity model and a user-facing prototype application, assessing both the technical performance and the social implications of such a system.

**1.3 Structure of This Document**

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

Inhalt

[1 Introduction ](#_Toc199212127)

1.1 Motivation

1.2 Research Question

1.3 Structure of This Document

[2 Related Work ](#_Toc199212129)

2.1 Traditional Approaches to Fake News Detection

2.2 Political Tweets and Misinformation

2.3 Multilingual and Cross-Lingual Models

2.4 Gaps and Novelty of Our Approach

[3 Methodology ](#_Toc199212130)

[3.1 General Methodology ](#_Toc199212131)

[3.2 Data Understanding and Preparation ](#_Toc199212132)

[3.3 Modeling and Evaluation ](#_Toc199212133)

[4 Results ](#_Toc199212134)

[5 Discussion ](#_Toc199212135)

5.1 Project Outcomes and Contributions

5.2 Limitations

5.3 Ethical Considerations

5.4 Future Research Directions

# 2 Related Work

Fake news detection, particularly in political contexts, has become a critical area of research due to the growing impact of false information on public discourse. Most previous work has focused on binary classification and the use of large-scale language models for identifying the accuracy of claims, but none have directly implemented a paraphrase-based, three-way classification model on political tweets using multilingual techniques. This highlights a new contribution of our project.

**2.1 Traditional Approaches to Fake News Detection**

Most existing studies classify content as either "fake" or "real." For example, Saeed & Al Solami (2023) explored multiple models—including SVM, CNN, LSTM, and BERT—for fake news detection and achieved accuracy rates between 92% and 99.2%. Alghamdi et al. (2022) compared deep learning and classical methods, reporting F1 scores of up to 93.17% using RoBERTa.

However, these approaches are limited to binary classification and often focus solely on English-language content or domain-specific datasets like PolitiFact and LIAR.

**2.2** **Political Tweets and Misinformation**

Several studies have applied machine learning to political tweets, but not in the way our model does. Rahim (2021) used a BERT-based approach to detect rumors during the US elections, highlighting challenges with partial truths and the complexity of annotation. Dadkhah et al. (2024) created a massive Twitter dataset (TruthSeeker) with over 180,000 labelled samples for real/fake detection, but again limited the analysis to binary outcomes.

Ornstein et al. (2025) applied large language models like GPT-3 and GPT-4 for political text analysis (sentiment, scaling, topic), but not for veracity classification. Their multi-class classification focuses on content types, not truthfulness.

**2.3** **Multilingual and Cross-Lingual Models**

Our project is particularly notable by its use of a multilingual paraphrase model (paraphrase-multilingual-MiniLM-L12-v2). In contrast, other studies like Kar et al. (2020) and Kazemi et al. (2022) employed multilingual transformer models such as mBERT, XLM-RoBERTa, and LaBSE for tasks like zero-shot learning or fact-check matching across languages. Kazemi et al. reported 86% accuracy in cross-lingual fact-check matching, demonstrating the possibility of multilingual applications but without three-class output.

**2.4** **Gaps and Novelty of Our Approach**

To date, no known study has used a paraphrase-based embedding model to compare political tweets with a fake news dataset and then classify them into three categories: true, false, and uncertain. Most prior research:

- Relies on binary classification (true/false).
- Uses monolingual data, primarily in English.
- Lacks semantic similarity-based comparison across datasets.

By combining semantic similarity, multilingual capability, and three-tier classification, our project introduces a new methodological model for assessing political content and misinformation.

# 3 Methodology

## 3.1 General Methodology

**3.1 General Methodology**

Our goal was to semantically assess political tweets by comparing them with a known fake news dataset and categorizing each tweet as true, false, or uncertain. To do this, we built a two-part system:

1. A backend pipeline for cleaning and preparing large-scale tweet data
2. A frontend interactive app for users to manually test statements using text similarity detection (TF-IDF + cosine similarity).

We structured our approach as follows:

1. Data Collection  
    We gathered two datasets:
    - A. jl-based dataset containing raw tweets from various ‘Bundestag’ members
    - A fake news dataset containing labelled entries (true or false)
2. Automated Preprocessing of Political Tweets  
    Using a custom Python script (datacleaning.py) we performed the following:
    - Parsed. jl files and extracted relevant fields (text, created_at, username)
    - Cleaned each tweet using regex patterns:
        - Removed emojis, mentions, hashtags, and URLs
        - Preserved ‚Umlaute‘ and ß
        - Lowercased the text
    - Filtered out empty entries
    - Saved results as .csv, splitting the file if it exceeded Excel’s row limit  
        → See code in _Appendix A_
3. Semantic Embedding & Similarity Analysis  
    We embedded the cleaned tweet dataset and the fake/true news dataset using the multilingual model paraphrase-multilingual-MiniLM-L12-v2.  
    For each tweet, we calculated the cosine similarity to both fake and true examples. Based on the scores:
    - True: If most similar to a verified example
    - False: If most similar to fake news
    - Unsure: If similarity scores were too low or inconclusive
4. Threshold-Based Three-Way Classification  
    We defined similarity thresholds to assign one of the three labels. A tweet was labelled unsure if no significant match to either category could be found.
5. Streamlit App Development (TF-IDF-Based Checker)  
    To make the system accessible, we built a Streamlit web app:
    - Allows users to upload fake news + tweet datasets
    - Uses TF-IDF vectorization and cosine similarity to check user-entered statements
    - Outputs a classification label (fake, real, or uncertain) based on the similarity score
    - Integrated explanations and visualization of similarity score  
        → See screenshot and example in _Section 5.2_
6. Evaluation  
    We manually reviewed output labels and performed basic accuracy checks on a subset of tweets. We also tested the app with known true and false claims.

## 3.2 Data Understanding and Preparation

Bundestag Twitter Data

<https://faubox.rrze.uni-erlangen.de/getlink/fi8W52QUEdtmm7LEGLiDBD/twitter-bundestag-2022.tar.gz>

Real & Fake News

<https://www.kaggle.com/datasets/razanaqvi14/real-and-fake-news>

- The Bundestag dataset contains a .jl file with tweets from Members of Parliament, where each item includes metadata and the tweet content itself.
- The Real & Fake News dataset consists of two CSV files: one with true news articles and the other with fake ones. Key attributes of this data include title, text, subject, and date of publication.
- A notable challenge is the difference in data formats and data richness between the two datasets.

## 3.3 Modeling and Evaluation

We decided on a bag of words approach built around TF-IDF features.The data preprocessing starts with removing unnecessary characters such as emojis, URLs, hashtags, mentions etc. Each tweet is reduced to text, lower-cased and stripped of German stop-words. The cleaned tweets are then merged with a fake-news collection to form a single corpus. A TF-IDF vectoriser is trained unsupervised on this joint corpus so that terms from both domains appear in the vocabulary. At inference time the incoming text is transformed with that same vectoriser and its vector is compared with the cosine similarity to all stored fake-news vectors. We interpret the score with simple thresholds: 0.70 or higher is classed as Fake, 0.30 or lower as Real, and anything in between as Uncertain. The entire system runs in one Streamlit app.For evaluation we prepare a labelled validation set containing both real and fake texts. We will sweep the similarity threshold to find the point that achieves the preferred balance.

# 4 Results

**4.1 Artifacts**

- App.py

App.by is both the Streamlit front-end and the model’s inference engine. The script defines the page title, icon and a single-screen layout where users can either upload a curated fake-news reference file plus a batch of tweets (CSV or XLSX) or paste a single statement into a text box. A helper routine detects whether each upload is CSV or Excel, loads the data with pandas and halts with a clear Streamlit error if the format is unsupported. It then loads German stop-words from the stop-words package, instantiates a TfidfVectorizer with 5 000 features, fits it on the cleaned fake-news texts and stores the resulting TF-IDF matrix as the reference. For every uploaded tweet or typed statement the script generates a TF-IDF vector, computes its cosine similarity to the reference matrix, keeps the highest score and assigns a label: scores ≥ 0.70 are Fake, scores ≤ 0.30 are Real, and anything in between is Uncertain. The interface displays the label, the similarity score to three decimal places and when the label is Fake an empty panel where an LLM could later insert an explanation.

- Datacleaning.py

Datacleaning.py can be seen as a preprocessing tool for turning raw Twitter data into a modelling-ready CSV. The script walks through every file and every tweet, applying its clean_text() routine: emojis are stripped using a broad Unicode range; URLs, hashtags, @-mentions and surplus punctuation are removed with concise regular expressions; German umlauts (ä, ö, ü) and “ß” are preserved; all text is lower-cased. For each tweet the cleaned text, user handle, tweet ID and timestamp are stored in memory and ultimately written to a single cleaned_tweets.csv.

- ML4B.ipynb

**4.2 Concept of the App**

The system is built on the following Python stack: Scikit-learn 1.7.0 supplies the TfidfVectorizer for feature extraction and cosine_similarity for distance measurement, while pandas 2.2.3 manages dataset loading, column cleaning, table joins and CSV/XLSX export. For language processing, NLTK 3.9.1 provides the standard German stop-word list, which we extend with additional social-media terms from the stop-words package (version 2018-07-23). Streamlit 1.45.1 serves as the web interface.The development takes place in Jupyter notebooks on Google Colab and in Visual Studio Code. All dependency versions are pinned in requirements.txt. The Streamlit application follows a simple workflow. First, the user is prompted to upload two files.A fake-news dataset and a general tweets dataset.Both files are required to be in CSV or XLSX format.Now the user is able to type in a statement.Once the statement is submitted, the app processes each text, computes the cosine-similarity score against the fake-news vectors, and returns a notification indicating whether the text is classified as Fake, Real or Uncertain, alongside with its similarity score.

# 5 Discussion

**5.1 Project Outcomes and Contributions**

Our project resulted in two key artifacts:

1. A semantic similarity-based classification system that labels political tweets as _true_, _false_, or _unsure_ using a multilingual transformer model.
2. An interactive Streamlit app that enables users to test custom input statements against a known fake news dataset using TF-IDF + cosine similarity.

This dual-system architecture allows both scalable, automated classification and user-friendly manual inspection. Our approach introduces a novel three-way classification paradigm in a domain that predominantly relies on binary decisions. Moreover, by using multilingual sentence embeddings, the system can be adapted to various languages and cultural contexts.

**5.2 Limitations**

Despite promising results, our approach and application face several limitations:

- Data Limitations:  
    The quality and representativeness of the fake news dataset significantly affect model performance. If the dataset lacks coverage of certain topics or styles, the model may struggle with semantic mismatches.
- Computational Constraints:  
    Our work was conducted using free Google Colab resources, which limited training times, batch sizes, and model complexity. Larger experiments (e.g., fine-tuning transformer models) were not feasible under these constraints.
- App Simplification:  
    The Streamlit app uses TF-IDF for practical reasons (speed, local interpretability), but lacks the deep semantic precision of transformer-based models. It also does not provide detailed justifications or counter-arguments—which would be important in real-world fact-checking.
- False Confidence in “Unsure” Results:  
    The “unsure” category relies on fixed thresholds. Some statements may seem uncertain due to lack of match—not because they are inherently ambiguous. This can mislead users if not clearly explained.

**5.3 Ethical Considerations**

Potential Dangers

- Misuse of Classifications:  
    A naïve application of the tool in political or journalistic contexts may result in oversimplified labelling. There is a risk of unintended censorship, especially if "false" labels are applied to tweets without context or explanation.
- Bias and Discrimination:  
    If the fake news dataset contains biases (e.g., more examples from certain political groups), the system may reflect or even amplify them. Multilingual models may also perform unevenly across languages, which could discriminate against non-English or low-resource contexts.
- Transparency & Explainability:  
    While we provide similarity scores and thresholds, the model does not explain why a statement is similar to fake news. This limits accountability and may influence user trust.
- Social Impact:  
    Algorithmic fact-checking can be helpful, but may also disregard alternative viewpoints, especially if implemented without proper oversight. Transparency and human review must remain part of the process.

Resources such as [AlgorithmWatch](https://algorithmwatch.org/en/) and [AI Now Institute](https://ainowinstitute.org/) emphasize these risks in their critical analyses of automated decision-making systems across Europe.

**5.4 Future Research Directions**

Our work opens several opportunities for future research:

- Improved Fact-Alignment Models:  
    Future systems could use retrieval-based methods (e.g., fact-check databases like Snopes, PolitiFact) and entailment modelling to assess factual accuracy more precisely.
- Context-Aware Classification:  
    Incorporating context from user history, event timelines, or quote-retweet networks may help better interpret a tweet's intent and truth value.
- Interactive Counterfactual Explanations:  
    Systems could provide "Why?" and "What if?" explanations—e.g., showing what a user could change in a sentence to make it more truthful or less misleading.
- Crowdsourced Evaluation:  
    Including user feedback and crowdsourced labels could help validate or adjust model classifications over time.
- Research Question Suggestions:
  - Can we train transformer models to detect “manipulative framing” rather than outright falsehoods?
  - How can we ensure semantic similarity models avoid cultural or political bias across languages?
  - What role should automated systems play in journalism and democratic accountability?
