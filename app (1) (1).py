import streamlit as st
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# --------------------------------
# TITLE SECTION
# --------------------------------
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center;'>Detect whether a news headline is <b>Real</b> or <b>Fake</b> using NLP (TF-IDF + Logistic Regression)</p>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fake_news.csv")
    df = df.dropna()
    return df

df = load_data()

# --------------------------------
# TEXT CLEANING
# --------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)

# --------------------------------
# TRAIN MODEL
# --------------------------------
@st.cache_resource
def train_model():
    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vectorized = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vectorized, y)

    return model, vectorizer

model, vectorizer = train_model()

# --------------------------------
# USER INPUT
# --------------------------------
st.subheader("🔍 Enter News Headline")

user_input = st.text_area("", height=120, placeholder="Type or paste news headline here...")

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter some news text.")
    else:
        cleaned = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        confidence = max(probabilities) * 100

        st.divider()

        if prediction == "fake":
            st.markdown(
                "<h2 style='color: red;'>❌ This appears to be FAKE news</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color: green;'>✅ This appears to be REAL news</h2>",
                unsafe_allow_html=True
            )

        st.progress(int(confidence))
        st.write(f"Confidence Score: {confidence:.2f}%")

# --------------------------------
# FOOTER
# --------------------------------
st.divider()
st.markdown(
    "<small>Model: TF-IDF + Logistic Regression | Dataset: 100 samples | Developed for NLP Practical</small>",
    unsafe_allow_html=True
)
