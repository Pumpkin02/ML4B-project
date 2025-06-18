import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words

# 页面配置
st.set_page_config(page_title="Fake News Checker", page_icon="🧠")
st.title("🧠 Fake-News Erkennung")
st.markdown(
    "Gib unten eine Aussage oder einen Tweet ein. "
    "Das System prüft per TF-IDF + Cosine Similarity, "
    "ob der Text bekannten Fake News ähnelt und gibt ein Ergebnis."
)

# —— 侧边栏：上传假新闻和推文文件（支持 csv 和 xlsx） ——
st.sidebar.header("Daten hochladen")
fake_file = st.sidebar.file_uploader(
    "Fake-News Datei (CSV oder XLSX)", type=['csv','xlsx']
)
train_file = st.sidebar.file_uploader(
    "Tweets Datei (CSV oder XLSX)", type=['csv','xlsx']
)

if not fake_file or not train_file:
    st.warning("Bitte lade zuerst sowohl die Fake-News als auch die Tweets Datei hoch!")
    st.stop()

# —— 根据后缀自动读取 —— 
def load_df(file):
    name = file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file)
    elif name.endswith(('.xls','.xlsx')):
        return pd.read_excel(file, engine='openpyxl')
    else:
        st.error("Unbekanntes Dateiformat: " + name)
        st.stop()

df_fake  = load_df(fake_file)
df_train = load_df(train_file)

# 文本预处理
df_fake ['text_clean'] = df_fake ['text']
df_train['text_clean'] = df_train['text']

# 丢弃空值
df_fake = df_fake.dropna(subset=['text_clean'])   
df_train = df_train.dropna(subset=['text_clean'])   

# Combine the corpora for fitting the TF-IDF vectorizer
combined_texts = df_fake['text'].tolist() + df_train['text_clean'].tolist()



# Load German stopwords via nltk

# —— 取出德语停用词列表 —— 
german_stopwords = get_stop_words('german')


# TF-IDF 矢量化（德语停用词）
vectorizer = TfidfVectorizer(
    stop_words=german_stopwords,
    max_features=5000
)
fake_tfidf = vectorizer.fit_transform(df_fake['text_clean'].tolist())

# 用户输入
user_input = st.text_area(
    "📝 Deine Aussage oder Tweet",
    placeholder="z. B. 'Die Erde ist flach.'",
    height=150
)

# 按钮触发
if st.button("🔍 Prüfen"):
    text = user_input.strip()
    if not text:
        st.warning("Bitte gib einen Text ein.")
    else:
        # 1. 对输入文本做 TF-IDF
        input_tfidf = vectorizer.transform([text])
        # 2. 计算余弦相似度并取最大
        sim_scores = cosine_similarity(input_tfidf, fake_tfidf)
        max_score  = sim_scores.max()
        # 3. 根据阈值打标签
        if max_score >= 0.7:
            label = "❌ Fake News"
        elif max_score <= 0.3:
            label = "✅ Echte News"
        else:
            label = "❓ Unklar"
        # 4. 显示结果
        st.markdown(f"### Ergebnis: {label}")
        st.write(f"Ähnlichkeitswert: {max_score:.3f}")
        # 5. 假新闻时的解释占位
        if label.startswith("❌"):
            st.markdown("### 🧾 Erklärung")
            st.info(
                "Der Text ähnelt bekannten Fake News. "
                "Hier könnte später ein LLM die genaue Falschbehauptung erläutern."
            )
