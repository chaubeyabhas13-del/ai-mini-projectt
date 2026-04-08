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
    st.subheader("💼 Smart Resume Analyzer")

    uploaded_file = st.file_uploader("📂 Upload Resume (TXT)", type=["txt"])
    job = st.text_area("📝 Enter Job Requirements (skills separated by space)")

    if st.button("🔍 Analyze Resume"):
        if uploaded_file is not None and job:

            # Read resume
            resume = uploaded_file.read().decode("utf-8").lower()
            job = job.lower()

            resume_words = resume.split()
            job_words = job.split()

            # Matching logic
            matched = [word for word in job_words if word in resume_words]
            missing = [word for word in job_words if word not in resume_words]

            score = len(matched)
            match_percent = int((score / len(job_words)) * 100)

            # UI OUTPUT
            st.markdown("### 📊 Analysis Dashboard")

            # Progress
            st.progress(match_percent / 100)

            # Score + Rating
            if match_percent >= 80:
                rating = "🌟 Excellent"
            elif match_percent >= 60:
                rating = "👍 Good"
            elif match_percent >= 40:
                rating = "⚠️ Average"
            else:
                rating = "❌ Poor"

            st.markdown(f"### 🎯 Match Score: {match_percent}%")
            st.markdown(f"### ⭐ Rating: {rating}")

            # Cards
            colA, colB = st.columns(2)

            with colA:
                st.success(f"✅ Matched Skills:\n{matched}")

            with colB:
                st.warning(f"❗ Missing Skills:\n{missing}")

            # Decision
            st.markdown("---")
            if match_percent >= 50:
                st.success("🎉 Final Result: Candidate Selected")
            else:
                st.error("❌ Final Result: Candidate Rejected")

        else:
            st.warning("⚠️ Upload resume and enter job requirements")
