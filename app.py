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
#dfdsaf
# Datenvisualisierung
st.subheader("Beispiel-Diagramm")
st.line_chart(data.set_index('Datum'))

print(123)
