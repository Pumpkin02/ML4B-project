import streamlit as st
import pandas as pd
import numpy as np

# Titel der App
st.title("Meine erste Streamlit App")

# Seitenleiste
st.sidebar.header("Einstellungen")
user_name = st.sidebar.text_input("Ihr Name", "Gast")

# Hauptbereich
st.write(f"Willkommen, {user_name}!")

# Beispiel-Daten
data = pd.DataFrame({
    'Datum': pd.date_range('2024-01-01', periods=10),
    'Werte': np.random.randn(10).cumsum()
})

# Datenvisualisierung
st.subheader("Beispiel-Diagramm")
st.line_chart(data.set_index('Datum'))

# Interaktive Elemente
st.subheader("Interaktive Elemente")
number = st.slider("Wählen Sie eine Zahl", 0, 100, 50)
st.write(f"Sie haben die Zahl {number} ausgewählt")

# Datei-Upload
st.subheader("Datei-Upload")
uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau der Daten:")
    st.dataframe(df.head()) 