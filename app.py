import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words

# Page configuration
st.set_page_config(page_title="Fake News Checker", page_icon="🧠")
st.title("🧠 Fake-News Detection")
st.markdown(
    "Enter a statement or tweet below. "
    "The system will use TF-IDF + cosine similarity "
    "to check whether the text resembles known fake news."
)

# Sidebar: upload fake news and tweets files (supports CSV and XLSX)
st.sidebar.header("Upload Data")
fake_file = st.sidebar.file_uploader(
    "Fake News File (CSV or XLSX)", type=['csv', 'xlsx']
)
train_file = st.sidebar.file_uploader(
    "Tweets File (CSV or XLSX)", type=['csv', 'xlsx']
)

if not fake_file or not train_file:
    st.warning("Please upload both the fake news file and the tweets file first!")
    st.stop()

# Automatically load dataframe based on file extension
def load_df(file):
    name = file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file)
    elif name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file, engine='openpyxl')
    else:
        st.error("Unknown file format: " + name)
        st.stop()

df_fake  = load_df(fake_file)
df_train = load_df(train_file)

# Text preprocessing
df_fake ['text_clean'] = df_fake ['text']
df_train['text_clean'] = df_train['text']

# Drop rows with missing text
df_fake  = df_fake.dropna(subset=['text_clean'])
df_train = df_train.dropna(subset=['text_clean'])

# Combine the corpora for fitting the TF-IDF vectorizer
combined_texts = df_fake['text_clean'].tolist() + df_train['text_clean'].tolist()

# Load German stopwords via stop-words package
german_stopwords = get_stop_words('german')

# TF-IDF vectorization (with German stopwords)
vectorizer = TfidfVectorizer(
    stop_words=german_stopwords,
    max_features=5000
)
fake_tfidf = vectorizer.fit_transform(df_fake['text_clean'].tolist())

# User input
user_input = st.text_area(
    "📝 Your statement or tweet",
    placeholder="e.g. 'Die Erde ist flach.'",
    height=150
)

# Button trigger
if st.button("🔍 Check"):
    text = user_input.strip()
    if not text:
        st.warning("Please enter some text.")
    else:
        # Apply TF-IDF to the input text
        input_tfidf = vectorizer.transform([text])
        
        # Compute cosine similarity and take the maximum score
        sim_scores = cosine_similarity(input_tfidf, fake_tfidf)
        max_score  = sim_scores.max()

        # Assign label based on threshold
        if max_score >= 0.7:
            label = "❌ Fake News"
        elif max_score <= 0.3:
            label = "✅ Real News"
        else:
            label = "❓ Uncertain"

        # Display the result
        st.markdown(f"### Result: {label}")
        st.write(f"Similarity score: {max_score:.3f}")

        # Placeholder explanation for fake news
        if label.startswith("❌"):
            st.markdown("### 🧾 Explanation")
            st.info(
                "The text resembles known fake news. "
                "An LLM could later provide a detailed explanation of the false claim."
            )
