import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page setup
st.set_page_config(page_title="AI Mini Project", layout="centered")

# Background + Style
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #eef2f3, #8e9eab);
}
h1 {
    text-align: center;
    color: #1f2937;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🧠 AI Powered System</h1>", unsafe_allow_html=True)
st.markdown("### Fake News Detection & Resume Screening")
st.markdown("---")

# Model
data = {
    "text": [
        "Government passes new law",
        "Aliens landed in India",
        "Economy is improving",
        "Fake cure for diseases"
    ],
    "label": [1, 0, 1, 0]
}

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
model = MultinomialNB()
model.fit(X, df["label"])

# Layout
col1, col2 = st.columns(2)

# Fake News Section
with col1:
    st.subheader("📰 Fake News Detection")
    news = st.text_area("Enter News Here")

    if st.button("Check News"):
        if news:
            pred = model.predict(vectorizer.transform([news]))
            if pred[0] == 1:
                st.success("✅ Real News")
            else:
                st.error("❌ Fake News")
        else:
            st.warning("Please enter news")

# Resume Section
with col2:
    st.subheader("💼 Resume Screening")
    resume = st.text_area("Resume Skills")
    job = st.text_area("Job Requirements")

    if st.button("Check Resume"):
        if resume and job:
            resume_words = resume.lower().split()
            job_words = job.lower().split()
            score = sum(1 for word in job_words if word in resume_words)

            st.info(f"Matching Score: {score}")

            if score >= len(job_words)//2:
                st.success("✅ Candidate Selected")
            else:
                st.error("❌ Candidate Rejected")
        else:
            st.warning("Enter both fields")

st.markdown("---")
st.caption("Developed by Abhas Chaubey")
