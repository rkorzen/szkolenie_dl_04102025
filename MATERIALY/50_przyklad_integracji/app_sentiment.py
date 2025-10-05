# app_sentiment.py
import streamlit as st
import torch
import torch.nn as nn
import joblib
from train_sentiment import SentimentNet

vectorizer = joblib.load("vectorizer.pkl")
model = SentimentNet(len(vectorizer.get_feature_names_out()), num_classes=2)
model.load_state_dict(torch.load("sentiment_model.pth"))
model.eval()

st.title("💬 Analiza sentymentu komentarzy")
st.write("Wpisz komentarz, a model oceni czy jest **pozytywny** czy **negatywny**.")

comment = st.text_area("Twój komentarz")

if st.button("Oceń"):
    if comment.strip():
        X = vectorizer.transform([comment]).toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred = model(X_tensor).argmax(dim=1).item()
        if pred == 1:
            st.success("👍 Pozytywny komentarz")
        else:
            st.error("👎 Negatywny komentarz")
    else:
        st.warning("Wpisz jakiś komentarz!")
