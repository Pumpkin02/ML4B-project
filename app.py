import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import time
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
    body {
        background-color: white;
        color: black;
    }
    
    
    .stApp {
        background-color: white;
        color: black;
    }
   
    /* Button 样式 */
    button[kind="secondary"] {
    background-color: #1e1e1e !important;
    color: white !important;
    border: 1px solid #ccc !important;
    border-radius: 5px;
    padding: 0.5em 1em;
    }
    button[kind="secondary"]:hover {
    background-color: #666666 !important;
    }

    /* Selectbox 标签字体颜色设置为黑色 */
    label {
    color: black !important;
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

# 先放一个空 progress 占位
#progress = st.empty()

# ----------------------
# Mothod description
# ----------------------

st.markdown("### Method Comparison: Mean Similarity vs KNN Voting")

st.markdown(
    """
    <style>
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }
    th, td {
        border: 1px solid #cccccc;
        text-align: left;
        padding: 8px;
    }
    th {
        background-color: #444444;  /* 深灰表头背景 */
        color: white;               /* 白色字体确保可读性 */
    }
    </style>

    <table>
        <tr>
            <th>Aspect</th>
            <th>Assessment of Mean Similarity</th>
            <th>KNN Voting (Top-K Similarity)</th>
        </tr>
        <tr>
            <td>Core Idea</td>
            <td>Compare the tweet to all fake and true news, then assign the label based on which group has higher average similarity.</td>
            <td>Find the K most similar news articles and vote among their labels to predict the tweet label.</td>
        </tr>
        <tr>
            <td>Computation</td>
            <td>Fast: only two cosine similarity batches per tweet</td>
            <td>Moderate: full similarity + sort per tweet</td>
        </tr>
        <tr>
            <td>Interpretability</td>
            <td>Moderate: based on group-level closeness</td>
            <td>High: most similar news examples can be shown</td>
        </tr>
        <tr>
            <td>Handling ambiguity</td>
            <td>Controlled by similarity difference threshold</td>
            <td>Controlled by vote margin (e.g. 3 vs 2 = unclear)</td>
        </tr>
        <tr>
            <td>Best for</td>
            <td>Quick overall estimation</td>
            <td>Detailed case-by-case judgment</td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("File must contain a 'text' column.")
        st.stop()

    # Preprocess text (light)
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)
        return re.sub(r"\s+", " ", text).strip()

    #df['text_clean'] = df['text'].astype(str).apply(clean_text)

    # Embed tweets
    #with st.spinner("Embedding tweets with multilingual BERT..."):
    #    tweet_embeddings = model.encode(df['text_clean'].tolist(), batch_size=64, show_progress_bar=True)

    texts = df['text_clean'].tolist()
    tweet_embeddings = []

    progress = st.progress(0, text="Embedding tweets...")

    for i, chunk in enumerate(range(0, len(texts), 64)):
        batch = texts[chunk:chunk+64]
        tweet_embeddings.extend(model.encode(batch))
        progress.progress(min((i+1)*64/len(texts), 1.0))

    progress.empty()  # 清除进度条

    done_message = st.empty()
    done_message.markdown(
    "<div style='padding:10px; border-radius:5px; background-color:#e6f4ea; "
    "border-left:5px solid #28a745; color:black;'>"
    "Embedding complete."
    "</div>",
    unsafe_allow_html=True
)
    time.sleep(2)
    done_message.empty()



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

    #st.markdown("### Assessment of Mean Similarity")
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

    #st.markdown("### KNN Voting based on Cosine Similarity")
    knn_labels = knn_voting(tweet_embeddings, news_embeddings, news_labels, news_texts)
    df['knn_label'] = knn_labels


    # ----------------------
    # Interactable DataTable
    # ----------------------

    st.markdown("### Result")
    selected_label_view = st.selectbox("Select method for display:", ['mean_label', 'knn_label'])
    st.dataframe(df[['text_clean', selected_label_view, 'sim_fake', 'sim_true']].sort_values(by=selected_label_view))

    # ----------------------
    # Most/Fewest Fake News Publisher
    # ----------------------
        
    def success_box(message: str):
        st.markdown(
            f"""
            <div style='
                padding: 10px;
                border-radius: 5px;
                background-color: #d4edda;
                border-left: 6px solid #28a745;
                color: black;
                font-size: 16px;
                margin-top: 10px;
            '>{message}</div>
            """, unsafe_allow_html=True
        )


        
    st.markdown("### Fake News Publishers")
    if 'user' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Most Fake News Publisher"):
                top_user = df[df['mean_label'] == 'Fake']['user'].value_counts().idxmax()
                #st.markdown(f"Most fake news comes from: **{top_user}**")
                success_box(f"Most fake news comes from: **{top_user}**")

        with col2:
            if st.button("Fewest Fake News Publisher"):
                user_counts = df[df['mean_label'] == 'Fake']['user'].value_counts()
                least_fake = user_counts[user_counts == user_counts.min()].index[0]
                #st.info(f"Least fake news: **{least_fake}**")
                success_box(f"Least fake news: **{least_fake}**")


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
        st.dataframe(filtered[['text_clean', 'mean_label', 'knn_label']])
else:
    st.markdown("### Result")
    st.markdown(
    """
    <div style='
        padding: 10px;
        border-radius: 5px;
        background-color: #2c3e50;
        border-left: 5px solid #2980b9;
        color: white;
        font-size: 15px;
        margin-top: 10px;
    '>
        Please upload a tweet dataset first to see predictions.
    </div>
    """,
    unsafe_allow_html=True
    )


