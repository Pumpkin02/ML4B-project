import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from stop_words import get_stop_words

# Seite konfigurieren
st.set_page_config(page_title="🧠 Fake-News Checker", page_icon="🧠", layout="centered")

# Design-Auswahl in Sidebar
st.sidebar.header("🎨 Einstellungen")
dark_mode = st.sidebar.toggle("🌙 Dark Mode aktivieren", value=False)

# CSS Dark/Light Theme
if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .css-1v0mbdj, .css-1d391kg, .stTextInput>div>div>input {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        .stTextArea textarea {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

# Titel & Beschreibung
st.title("🧠 Fake-News Detection mit Ähnlichkeitsanalyse")
st.markdown("""
Gib unten eine Aussage oder einen Tweet ein.  
Das System prüft mithilfe von **TF-IDF + Cosine Similarity**, ob der Text bekannten Fake-News ähnelt.
""")

# Seitenleiste – Dateiupload
st.sidebar.subheader("📁 Daten hochladen (CSV oder Excel)")
fake_file = st.sidebar.file_uploader("📰 Fake-News-Datei", type=['csv', 'xlsx'])
tweet_file = st.sidebar.file_uploader("🐦 Tweets-Datei", type=['csv', 'xlsx'])

# Funktion zum sicheren Laden der Datei
def load_df(file, name=""):
    if not file:
        return None
    filename = file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file, header=0)
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file, header=0, engine='openpyxl')
    else:
        st.error(f"❌ Unbekanntes Dateiformat: {filename}")
        return None
    df.columns = df.columns.str.strip().str.lower()
    st.write(f"🔎 Vorschau der Datei *{name}*:")
    st.dataframe(df.head(3))
    return df

# Daten laden
df_fake = load_df(fake_file, "Fake-News")
df_tweets = load_df(tweet_file, "Tweets")

if df_fake is None or df_tweets is None:
    st.warning("⚠️ Bitte lade **beide Dateien** hoch.")
    st.stop()

if 'text' not in df_fake.columns and 'text_clean' in df_fake.columns:
    df_fake['text'] = df_fake['text_clean']
if 'text' not in df_tweets.columns and 'text_clean' in df_tweets.columns:
    df_tweets['text'] = df_tweets['text_clean']

if 'text' not in df_fake.columns or 'text' not in df_tweets.columns:
    st.error("❌ Beide Dateien müssen eine Spalte namens `text` oder `text_clean` enthalten.")
    st.stop()

# Vorverarbeitung
def preprocess(df):
    df['text_clean'] = df['text'].astype(str).str.strip()
    df = df.dropna(subset=['text_clean'])
    df = df[df['text_clean'] != ""]
    return df

df_fake = preprocess(df_fake)
df_tweets = preprocess(df_tweets)

# Stopwords laden
stopwords = get_stop_words('german')

# TF-IDF vorbereiten & cachen
@st.cache_resource
def prepare_vectorizer_and_index(fake_texts, all_texts):
    vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000)
    vectorizer.fit(all_texts)
    tfidf_fake = vectorizer.transform(fake_texts)
    nn_index = NearestNeighbors(n_neighbors=20, metric='cosine')
    nn_index.fit(tfidf_fake)
    return vectorizer, tfidf_fake, nn_index

vectorizer, fake_tfidf, nn_index = prepare_vectorizer_and_index(
    df_fake['text_clean'].tolist(),
    df_fake['text_clean'].tolist() + df_tweets['text_clean'].tolist()
)

# Benutzereingabe
st.subheader("📝 Dein Text")
user_input = st.text_area("Was möchtest du prüfen?", placeholder="z. B. 'Die Erde ist flach.'", height=150)

if st.button("🔍 Fake-News prüfen"):
    text = user_input.strip()
    if not text:
        st.warning("⚠️ Bitte gib einen Text ein.")
    else:
        input_vec = vectorizer.transform([text])
        distances, indices = nn_index.kneighbors(input_vec, return_distance=True)
        top_scores = 1 - distances.flatten()
        max_score = top_scores.max()

        if max_score >= 0.7:
            result = "❌ Fake News"
            explanation = "Der Text ähnelt stark bekannten Falschmeldungen."
        elif max_score <= 0.3:
            result = "✅ Eher seriös"
            explanation = "Der Text unterscheidet sich deutlich von bekannten Fake-News."
        else:
            result = "❓ Unklar"
            explanation = "Der Text liegt im Graubereich – könnte teilweise Fake-News enthalten."

        st.markdown(f"## 🧾 Ergebnis: {result}")
        st.metric(label="Ähnlichkeitswert", value=f"{max_score:.3f}")
        st.info(explanation)

        if result.startswith("❌"):
            st.markdown("### 🤖 Erklärung durch KI")
            with st.spinner("Generiere eine Begründung..."):
                keywords = ["flach", "impfung", "klima", "5g", "chip", "geheim", "gates", "covid", "corona"]
                matched = [k for k in keywords if k.lower() in text.lower()]
                if matched:
                    keyword_str = ", ".join(matched)
                    gpt_reason = (
                        f"Der Text enthält Schlagworte wie **{keyword_str}**, "
                        f"die häufig in bekannten Verschwörungserzählungen oder Falschinformationen vorkommen. "
                        f"Daher liegt der Verdacht nahe, dass es sich um eine Fake-News handelt."
                    )
                else:
                    gpt_reason = (
                        "Der Text ähnelt stark bekannten Fake-News in Struktur oder Inhalt. "
                        "Eine genauere Analyse könnte mithilfe von Faktenchecks erfolgen."
                    )
                st.success(gpt_reason)
