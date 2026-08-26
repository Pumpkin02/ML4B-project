import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------
# SETUP
# ----------------------
st.set_page_config("Fake-News Detection", layout="wide")
st.title("Fake-News Detection")

st.caption(
    "Similarity-based classification of German political tweets against a labelled "
    "English news corpus (ISOT), using multilingual sentence embeddings. "
    "Tweets with low confidence are deliberately left as 'Unclear'."
)

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

    label {
    color: black !important;
    }
    </style>
    """
    st.markdown(light_style, unsafe_allow_html=True)


# ----------------------
# Load model & cached news embeddings
# ----------------------
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    with open("news_embeddings.pkl", "rb") as f:
        news_data = pickle.load(f)
    return (
        model,
        np.array(news_data['embeddings']),
        np.array(news_data['labels']),
        news_data['texts'],
    )


model, news_embeddings, news_labels, news_texts = load_model_and_embeddings()


# ----------------------
# Built-in demo data (fallback if data/demo_tweets.csv is missing)
# ----------------------
DEMO_ROWS = [
    ("user_a", "Die neuen Klimaziele der Regierung sind heute im Bundestag beschlossen worden. #Klimaschutz"),
    ("user_a", "Wir brauchen endlich eine ehrliche Debatte über die Finanzierung der Rente. #Rente"),
    ("user_b", "Angeblich soll die EU heimlich neue Steuern planen - davon steht nirgendwo etwas. #EU"),
    ("user_b", "Die Zahlen des Statistischen Bundesamtes zeigen einen Rückgang der Arbeitslosigkeit."),
    ("user_c", "Man erzählt uns seit Jahren Märchen über die angeblichen Vorteile dieser Politik. #Politik"),
    ("user_c", "Heute Besuch in einer Schule im Wahlkreis - danke für die spannenden Fragen!"),
    ("user_d", "Die Inflation ist im letzten Quartal erneut gesunken, das entlastet die Haushalte. #Wirtschaft"),
    ("user_d", "Niemand sagt Ihnen die Wahrheit über die wahren Hintergründe dieser Entscheidung."),
    ("user_e", "Der Ausschuss hat den Gesetzentwurf mit breiter Mehrheit angenommen. #Bundestag"),
    ("user_e", "Skandal! Was hier wirklich passiert, wird von den Medien komplett verschwiegen."),
    ("user_f", "Digitalisierung der Verwaltung kommt voran, aber deutlich zu langsam. #Digitalisierung"),
    ("user_f", "Wir haben heute über den Haushalt 2023 debattiert. Die Redebeiträge sind online."),
]


def load_demo_dataframe():
    demo_path = os.path.join("data", "demo_tweets.csv")
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path)
    return pd.DataFrame(DEMO_ROWS, columns=["user", "text"])


# ----------------------
# Input: upload or demo
# ----------------------
col_up, col_demo = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload tweet CSV file (must contain a 'text' column)", type=["csv"]
    )
with col_demo:
    st.write("")
    st.write("")
    demo_clicked = st.button("▶ Load demo data")

if "use_demo" not in st.session_state:
    st.session_state.use_demo = False
if demo_clicked:
    st.session_state.use_demo = True
if uploaded_file is not None:
    st.session_state.use_demo = False

df_input = None
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
elif st.session_state.use_demo:
    df_input = load_demo_dataframe()


# ----------------------
# Method description
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
        background-color: #444444;
        color: white;
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
    unsafe_allow_html=True,
)


if df_input is not None:
    df = df_input.copy()

    # ----------------------
    # Column handling
    # ----------------------
    if 'text' not in df.columns:
        candidates = [c for c in ['tweet', 'content', 'full_text', 'text_clean']
                      if c in df.columns]
        if candidates:
            df['text'] = df[candidates[0]].astype(str)
        else:
            st.error(
                "File must contain a 'text' column. "
                f"Columns found: {list(df.columns)}"
            )
            st.stop()

    # Preprocess text (light) -- this line used to be commented out, which caused
    # a KeyError on 'text_clean' further down.
    def clean_text(text):
        text = re.sub(r"http\S+", "", str(text))
        return re.sub(r"\s+", " ", text).strip()

    if 'text_clean' not in df.columns:
        df['text_clean'] = df['text'].astype(str).apply(clean_text)
    else:
        df['text_clean'] = df['text_clean'].astype(str)

    df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)
    if len(df) == 0:
        st.error("No usable rows after cleaning.")
        st.stop()

    # ----------------------
    # Embed tweets
    # ----------------------
    texts = df['text_clean'].tolist()
    tweet_embeddings = []

    progress = st.progress(0, text="Embedding tweets...")
    n_batches = max(1, int(np.ceil(len(texts) / 64)))
    for i, start in enumerate(range(0, len(texts), 64)):
        batch = texts[start:start + 64]
        tweet_embeddings.extend(model.encode(batch))
        progress.progress(min((i + 1) / n_batches, 1.0))
    progress.empty()

    tweet_embeddings = np.array(tweet_embeddings)

    done_message = st.empty()
    done_message.markdown(
        "<div style='padding:10px; border-radius:5px; background-color:#e6f4ea; "
        "border-left:5px solid #28a745; color:black;'>"
        "Embedding complete."
        "</div>",
        unsafe_allow_html=True,
    )
    time.sleep(1.5)
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
        labels = np.where(diff < threshold, 'Unclear',
                          np.where(sim_fake > sim_true, 'Fake', 'True'))
        return labels, sim_fake, sim_true

    mean_labels, sim_f, sim_t = mean_similarity_prediction(
        tweet_embeddings, news_embeddings, news_labels
    )
    df['mean_label'] = mean_labels
    df['sim_fake'] = sim_f
    df['sim_true'] = sim_t

    # ----------------------
    # KNN Voting
    # ----------------------
    def knn_voting(tweet_embs, news_embs, news_lbls, K=5):
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

    df['knn_label'] = knn_voting(tweet_embeddings, news_embeddings, news_labels)

    # ----------------------
    # Results
    # ----------------------
    st.markdown("### Result")

    c1, c2, c3 = st.columns(3)
    counts = df['mean_label'].value_counts()
    c1.metric("Tweets analysed", len(df))
    c2.metric("Flagged as Fake", int(counts.get('Fake', 0)))
    c3.metric("Left as Unclear", int(counts.get('Unclear', 0)))

    selected_label_view = st.selectbox(
        "Select method for display:", ['mean_label', 'knn_label']
    )
    st.dataframe(
        df[['text_clean', selected_label_view, 'sim_fake', 'sim_true']]
        .sort_values(by=selected_label_view)
    )

    st.download_button(
        "⬇ Download results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="fake_news_results.csv",
        mime="text/csv",
    )

    # ----------------------
    # Publishers
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
            """,
            unsafe_allow_html=True,
        )

    if 'user' in df.columns:
        st.markdown("### Fake News Publishers")
        fake_only = df[df['mean_label'] == 'Fake']
        if len(fake_only) == 0:
            st.info("No tweets were classified as 'Fake' in this dataset.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Most Fake News Publisher"):
                    top_user = fake_only['user'].value_counts().idxmax()
                    success_box(f"Most fake news comes from: **{top_user}**")
            with col2:
                if st.button("Fewest Fake News Publisher"):
                    user_counts = fake_only['user'].value_counts()
                    least_fake = user_counts[user_counts == user_counts.min()].index[0]
                    success_box(f"Least fake news: **{least_fake}**")

    # ----------------------
    # Hashtag Filter
    # ----------------------
    def extract_hashtags(text):
        return re.findall(r"#\w+", str(text))

    df['hashtags'] = df['text'].apply(extract_hashtags)
    all_tags = sorted(set(tag for tags in df['hashtags'] for tag in tags))

    if all_tags:
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
            Upload a tweet dataset (CSV with a 'text' column) or click
            <b>Load demo data</b> above to see the system in action.
        </div>
        """,
        unsafe_allow_html=True,
    )
