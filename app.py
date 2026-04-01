import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("AI Mini Project")

# Fake News Model
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

# Fake News UI
st.header("Fake News Detection")
news = st.text_input("Enter news")

if st.button("Check News"):
    if news:
        pred = model.predict(vectorizer.transform([news]))
        if pred[0] == 1:
            st.success("Real News")
        else:
            st.error("Fake News")

# Resume Screening
st.header("Resume Screening")
resume = st.text_input("Resume Skills")
job = st.text_input("Job Requirements")

if st.button("Check Resume"):
    if resume and job:
        resume_words = resume.lower().split()
        job_words = job.lower().split()
        score = sum(1 for word in job_words if word in resume_words)

        if score >= len(job_words)//2:
            st.success("Selected")
        else:
            st.error("Rejected")
