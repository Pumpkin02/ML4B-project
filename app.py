import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------
# SETUP
# ----------------------
st.set_page_config("Fake-News Detection", layout="wide")
st.title("Fake-News Detection")

# Sidebar theme switch
theme = st.sidebar.selectbox("🎨 Select Theme", ["Light", "Dark"])

# Apply CSS styles dynamically
if theme == "Dark":
    dark_style = """
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    table, th, td {
        color: white !important;
    }
    </style>
    """
    st.markdown(dark_style, unsafe_allow_html=True)
else:
    light_style = """
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    </style>
    """
    st.markdown(light_style, unsafe_allow_html=True)


# Load model & cached news embeddings
@st.cache_resource

def load_model_and_embeddings():
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    with open("news_embeddings.pkl", "rb") as f:
        news_data = pickle.load(f)
    return model, news_data['embeddings'], np.array(news_data['labels']), news_data['texts']

model, news_embeddings, news_labels, news_texts = load_model_and_embeddings()

# ----------------------
# Upload tweet data
# ----------------------

uploaded_file = st.file_uploader("Upload tweet CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("File must contain a 'text' column.")
        st.stop()

    # Preprocess text (light)
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)
        return re.sub(r"\s+", " ", text).strip()

    df['text_clean'] = df['text'].astype(str).apply(clean_text)

    # Embed tweets
    with st.spinner("Embedding tweets with multilingual BERT..."):
        tweet_embeddings = model.encode(df['text_clean'].tolist(), batch_size=64, show_progress_bar=True)

    # ----------------------
    # Mean Similarity Assessment
    # ----------------------

    def mean_similarity_prediction(tweet_embs, news_embs, news_lbls, threshold=0.01):
        fake_vecs = news_embs[news_lbls == 0]
        true_vecs = news_embs[news_lbls == 1]
        sim_fake = cosine_similarity(tweet_embs, fake_vecs).mean(axis=1)
        sim_true = cosine_similarity(tweet_embs, true_vecs).mean(axis=1)
        diff = np.abs(sim_fake - sim_true)
        labels = np.where(diff < threshold, 'Unclear', np.where(sim_fake > sim_true, 'Fake', 'True'))
        return labels, sim_fake, sim_true

    st.markdown("### Assessment of Mean Similarity")
    mean_labels, sim_f, sim_t = mean_similarity_prediction(tweet_embeddings, news_embeddings, news_labels)
    df['mean_label'] = mean_labels
    df['sim_fake'] = sim_f
    df['sim_true'] = sim_t

    # ----------------------
    # KNN Voting
    # ----------------------

    def knn_voting(tweet_embs, news_embs, news_lbls, news_texts, K=5):
        sim_matrix = cosine_similarity(tweet_embs, news_embs)
        top_k_idx = np.argsort(sim_matrix, axis=1)[:, -K:]
        pred_labels = []
        for indices in top_k_idx:
            top_labels = news_lbls[indices]
            votes_fake = (top_labels == 0).sum()
            votes_true = (top_labels == 1).sum()
            if abs(votes_fake - votes_true) < 2:
                pred = "Unclear"
            elif votes_fake > votes_true:
                pred = "Fake"
            else:
                pred = "True"
            pred_labels.append(pred)
        return pred_labels

    st.markdown("### KNN Voting based on Cosine Similarity")
    knn_labels = knn_voting(tweet_embeddings, news_embeddings, news_labels, news_texts)
    df['knn_label'] = knn_labels

    # ----------------------
    # Interactable DataTable
    # ----------------------

    st.markdown("### Dynamic Tweet Table")
    selected_label_view = st.selectbox("Select method for display:", ['mean_label', 'knn_label'])
    st.dataframe(df[['text', 'text_clean', selected_label_view, 'sim_fake', 'sim_true']].sort_values(by=selected_label_view))

    # ----------------------
    # Most/Fewest Fake News Publisher
    # ----------------------
    st.markdown("### Fake News Publishers")
    if 'user' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Most Fake News Publisher"):
                top_user = df[df['mean_label'] == 'Fake']['user'].value_counts().idxmax()
                st.success(f"Most fake news comes from: **{top_user}**")
        with col2:
            if st.button("Fewest Fake News Publisher"):
                user_counts = df[df['mean_label'] == 'Fake']['user'].value_counts()
                least_fake = user_counts[user_counts == user_counts.min()].index[0]
                st.info(f"Least fake news: **{least_fake}**")

    # ----------------------
    # Hashtag Filter
    # ----------------------
    def extract_hashtags(text):
        return re.findall(r"#\w+", text)

    df['hashtags'] = df['text'].apply(lambda x: extract_hashtags(str(x)))
    all_tags = sorted(set(tag for tags in df['hashtags'] for tag in tags))

    st.markdown("### Thema wählen")
    selected_hashtag = st.selectbox("Wähle ein Schlagwort:", options=all_tags)
    if selected_hashtag:
        filtered = df[df['hashtags'].apply(lambda tags: selected_hashtag in tags)]
        st.dataframe(filtered[['text', 'text_clean', 'mean_label', 'knn_label']])
