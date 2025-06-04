import json
import pandas as pd
from tqdm import tqdm
import os
import re

# Pfad zum Ordner mit den Rohdateien (.jl)
ordner = "afd_tweets_raw"
alle_dateien = [f for f in os.listdir(ordner) if f.endswith(".jl")]

# Textbereinigung mit Emoji-Entfernung, aber Erhalt von Umlauten
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Emojis entfernen (breiter Unicode-Bereich)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Smileys
        u"\U0001F300-\U0001F5FF"  # Symbole
        u"\U0001F680-\U0001F6FF"  # Transport
        u"\U0001F1E0-\U0001F1FF"  # Flaggen
        u"\U00002500-\U00002BEF"  # Kästchen etc.
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # URLs, Hashtags, Mentions entfernen
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # Sonderzeichen entfernen, aber Umlaute & ß behalten
    text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s]", "", text)

    return text.lower().strip()

# Liste für saubere Tweets
gesammelt = []

# Verarbeitung jeder Datei
for datei in tqdm(alle_dateien, desc="Verarbeite .jl-Dateien"):
    with open(os.path.join(ordner, datei), "r", encoding="utf-8") as f:
        for zeile in f:
            eintrag = json.loads(zeile)
            tweets = eintrag.get("response", {}).get("data", [])
            nutzer = eintrag.get("account_name", None)

            if not nutzer:
                continue

            for tweet in tweets:
                text = tweet.get("text", "")
                created_at = tweet.get("created_at", None)

                if not text or not created_at:
                    continue

                text_clean = clean_text(text)
                if not text_clean:
                    continue  # Leerer Text nach Bereinigung → raus

                gesammelt.append({
                    "username": nutzer,
                    "created_at": created_at,
                    "text_clean": text_clean
                })

# In DataFrame umwandeln
df = pd.DataFrame(gesammelt)

# Maximalzeilen für Excel-kompatible CSVs
max_zeilen_excel = 1_048_000

# Aufteilen oder speichern
if len(df) > max_zeilen_excel:
    print(f"⚠️ {len(df)} Zeilen – Datei wird in Teile aufgeteilt ...")
    for i in range(0, len(df), max_zeilen_excel):
        teil = df.iloc[i:i + max_zeilen_excel]
        dateiname = f"afd_tweets_bereinigt_teil_{i // max_zeilen_excel + 1}.csv"
        teil.to_csv(dateiname, index=False, encoding="utf-8")
        print(f"💾 Gespeichert: {dateiname}")
else:
    df.to_csv("afd_tweets_bereinigt.csv", index=False, encoding="utf-8")
    print("✅ Gespeichert: afd_tweets_bereinigt.csv")

