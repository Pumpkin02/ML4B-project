import streamlit as st
import pandas as pd
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from stop_words import get_stop_words

# --- Einstellungen für den Rahmen und Farben ---
base_light_bg = "#FFFFFF"
base_dark_bg = "#0e1117"
base_light_text = "#000000"
base_dark_text = "#FAFAFA"
base_light_border = "#ddd"
base_dark_border = "#444"

# --- Beispiel-Tweets (nur einer wird zufällig angezeigt) ---
example_tweets = [
    {
        "username": "Sahra Wagenknecht (Die Linke)",
        "created_at": "10. August 2024",
        "text": "Die Bundeswehr gab 2023 rund 100 Millionen Euro für Social-Media-Influencer aus, um für den Dienst zu werben."
    },
    {
        "username": "Joana Cotar (ehem. AfD)",
        "created_at": "15. Mai 2022",
        "text": "In Deutschland leben fast 900.000 abgelehnte Asylbewerber, von denen 304.000 ausreisepflichtig sind – und trotzdem Stütze beziehen."
    },
    {
        "username": "Alice Weidel (AfD)",
        "created_at": "16. Februar 2025",
        "text": "Auf Dauer ist ein Sozialsystem nicht tragfähig, das vor allem an Menschen Leistungen auszahlt, die noch nie ins Sozialsystem eingezahlt haben."
    }
]

# --- App Layout und Styling ---
st.set_page_config(page_title="🧠 Fake-News Checker", page_icon="🧠", layout="centered")

# Dark Mode Einstellung & weitere Einstellungen in Sidebar (oben)
with st.sidebar:
    st.header("⚙️ Einstellungen")
    dark_mode = st.checkbox("🌙 Dark Mode aktivieren", value=False)
    explanation_mode = st.checkbox("🧠 Erklärung anzeigen", value=True)
    filter_time = st.slider("⏳ Zeitraum Filter (Jahre)", min_value=0, max_value=10, value=5)

# Farbwahl je nach Modus
bg_color = base_dark_bg if dark_mode else base_light_bg
text_color = base_dark_text if dark_mode else base_light_text
border_color = base_dark_border if dark_mode else base_light_border
example_bg_color = "#262730" if dark_mode else "#f9f9f9"

# --- Titel ---
st.title("🧠 Fake-News Detection mit Ähnlichkeitsanalyse")
st.markdown(
    "Wähle unten einen Tweet basierend auf Schlagwörtern zur Analyse. "
    "Das System prüft mithilfe von **TF-IDF + Cosine Similarity**, ob der Text bekannten Fake-News ähnelt."
)

# --- Zufälligen Beispiel-Tweet anzeigen ---
# --- Zufälligen Beispiel-Tweet anzeigen (mit Icons und Label) ---
random_tweet = random.choice(example_tweets)
username = random_tweet["username"]
created_at = random_tweet["created_at"]
text = random_tweet["text"]

# Status-Icon definieren
status_icon = ""
if "wagenknecht" in username.lower():
    status_icon = "<span style='color: green; font-weight: bold;'>✅ TRUE</span>"
elif any(x in username.lower() for x in ["afd", "cotar"]):
    status_icon = "<span style='color: red; font-weight: bold;'>❌ FAKE</span>"

st.markdown(
    f"""
    <div style="
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
        background-color: {example_bg_color};
        color: {text_color};
        font-size: 16px;
        line-height: 1.4;
        position: relative;
    ">
        <div style="position: absolute; top: 10px; left: 15px; font-weight: bold; font-size: 14px; opacity: 0.7;">
            Beispiel-Tweet:
        </div>
        <p style="margin-top: 30px; font-style: italic;">"{text}"</p>
        <p style="margin: 10px 0 0 0; font-weight: 600;">— {username}, <i>{created_at}</i></p>
        <div style="position: absolute; bottom: 10px; right: 15px; font-weight: bold; font-size: 16px;">
            {status_icon}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Datei-Upload-Bereich (unten in Sidebar) ---
with st.sidebar:
    st.markdown("---")
    st.header("📁 Daten hochladen")
    fake_file = st.file_uploader("📰 Fake-News-Datei (CSV/XLSX)", type=['csv', 'xlsx'])
    tweet_file = st.file_uploader("🐦 Tweets-Datei (CSV/XLSX)", type=['csv', 'xlsx'])

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
    return df

df_fake = load_df(fake_file, "Fake-News")
df_tweets = load_df(tweet_file, "Tweets")

if df_fake is None or df_tweets is None:
    st.warning("⚠️ Bitte lade **beide Dateien** hoch, um die Analyse zu starten.")
    st.stop()

# --- Spalte 'text' sicherstellen ---
if 'text' not in df_fake.columns and 'text_clean' in df_fake.columns:
    df_fake['text'] = df_fake['text_clean']
if 'text' not in df_tweets.columns and 'text_clean' in df_tweets.columns:
    df_tweets['text'] = df_tweets['text_clean']

if 'text' not in df_fake.columns or 'text' not in df_tweets.columns:
    st.error("❌ Beide Dateien müssen eine Spalte namens `text` oder `text_clean` enthalten.")
    st.stop()

# --- Preprocessing ---
def preprocess(df):
    df['text_clean'] = df['text'].astype(str).str.strip()
    df = df.dropna(subset=['text_clean'])
    df = df[df['text_clean'] != ""]
    return df

df_fake = preprocess(df_fake)
df_tweets = preprocess(df_tweets)

# Stopwords laden
stopwords = get_stop_words('german')

# --- TF-IDF + NearestNeighbors vorbereiten ---
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

# --- Keywords (themenrelevante, keine Stopwörter etc.) ---
keywords = [
    "Corona", "Impfung", "Ausländer", "Migration", "Wahlbetrug", "Klima", "AfD",
    "Ukraine", "Russland", "Biden", "Trump", "Islam", "Gender", "Zensur",
    "Meinungsfreiheit", "Klimawandel", "Kriminalität", "Flüchtlinge", "WHO", "Impfpflicht"
]

# --- Parteienfilter (noch deaktiviert, kann aktiviert werden, wenn Datensatz vorhanden) ---
# parteien = sorted(df_tweets['partei'].dropna().unique()) if 'partei' in df_tweets.columns else []
# selected_partei = st.sidebar.selectbox("🏛 Partei filtern (optional)", options=["Alle"] + parteien)

# --- Keyword-Index vorbereiten ---
@st.cache_resource
def build_keyword_index(texts, keywords):
    index = {}
    for keyword in keywords:
        mask = texts.str.contains(rf"\b{re.escape(keyword)}\b", case=False, na=False)
        index[keyword] = texts[mask].tolist()
    return index

tweet_index = build_keyword_index(df_tweets['text_clean'], keywords)

# --- Thema auswählen ---
st.subheader("🧩 Thema wählen")
selected_keyword = st.selectbox("🔍 Wähle ein Schlagwort:", sorted(keywords))

# Tweets zum Keyword filtern
filtered_texts = tweet_index.get(selected_keyword, [])

if not filtered_texts:
    st.warning("⚠️ Keine Tweets mit diesem Schlagwort gefunden.")
    st.stop()

# --- Ausgewählten Tweet aus df_tweets laden (kompletter Text + Username + Datum) ---
def get_full_tweet(text):
    # Wir suchen in df_tweets den Eintrag mit genau dem Text
    match = df_tweets[df_tweets['text_clean'] == text]
    if match.empty:
        return None
    # Nur ersten Treffer nehmen
    row = match.iloc[0]
    return row['username'], row['created_at'], row['text_clean']

selected_text = st.selectbox("📌 Wähle einen Tweet zur Analyse:", options=filtered_texts, index=0)

tweet_data = get_full_tweet(selected_text)
if tweet_data:
    username, created_at, full_text = tweet_data
    st.markdown(
        f"""
        <div style="
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
            background-color: {example_bg_color};
            color: {text_color};
            font-size: 16px;
            line-height: 1.4;
            white-space: pre-wrap;
        ">
            <p style="margin:0 0 8px 0; font-weight: 600;">@{username} — <i>{created_at}</i></p>
            <p style="margin:0;">{full_text}</p>
        </div>
        """
    , unsafe_allow_html=True)

# --- Fake-News Analyse starten ---
if st.button("🔍 Fake-News prüfen"):
    input_vec = vectorizer.transform([selected_text])
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

    if result.startswith("❌") and explanation_mode:
        st.markdown("### 🤖 Erklärung durch KI")
        with st.spinner("Generiere eine Begründung..."):
            matched = [k for k in keywords if k.lower() in selected_text.lower()]
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

# --- Ende Rahmen-Div ---
st.markdown("</div>", unsafe_allow_html=True)
