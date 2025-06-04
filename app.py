import streamlit as st

# === Seitentitel und Beschreibung ===
st.set_page_config(page_title="Fake News Checker", page_icon="🧠")
st.title("🧠 Fake-News Erkennung")
st.markdown("Gib unten eine Aussage oder einen Tweet ein. Das System prüft später mithilfe eines KI-Modells, ob es sich um Fake News handelt – und erklärt ggf. den wahren Hintergrund.")

# === Texteingabe ===
user_input = st.text_area("📝 Deine Aussage oder Tweet", placeholder="z. B. 'Die Erde ist flach.'", height=150)

# === Analyse starten ===
if st.button("🔍 Prüfen"):
    if not user_input.strip():
        st.warning("Bitte gib einen Text ein.")
    else:
        # === Platzhalter: KI-Modell zur Fake-News-Erkennung ===
        # Ersetze dies später mit eurem Klassifikator (z. B. model.predict(...))
        st.info("📌 Klassifikator läuft hier bald...")
        fake_or_real = "❓ Noch kein Ergebnis – KI-Modell fehlt"

        # === Platzhalter-Ergebnis anzeigen ===
        st.markdown(f"### Ergebnis: {fake_or_real}")

        # === Falls Fake – Erklärung generieren (Platzhalter) ===
        if fake_or_real.startswith("❌"):
            # Später z. B. mit GPT oder LLM ersetzen
            st.markdown("### 🧾 Erklärung")
            st.info("📌 Hier erscheint später eine automatisch generierte Erklärung, warum das Fake ist.")
