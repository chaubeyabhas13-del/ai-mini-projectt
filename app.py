import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Mini Project", layout="centered")

# Styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #eef2f3, #8e9eab);
}
h1 {
    text-align: center;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1> AI Powered System</h1>", unsafe_allow_html=True)
st.markdown("### Fake News Detection & Resume Screening")

# Tabs
tab1, tab2 = st.tabs([" Fake News", " Resume Analyzer"])

# -------------------------
# Fake News
# -------------------------
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

with tab1:
    st.subheader("📰 Fake News Detection")
    news = st.text_area("Enter News")

    if st.button("Check News"):
        if news:
            pred = model.predict(vectorizer.transform([news]))
            if pred[0] == 1:
                st.success("✅ Real News")
            else:
                st.error("❌ Fake News")
        else:
            st.warning("Enter news first")

# -------------------------
# Resume Analyzer
# -------------------------
with tab2:
    st.subheader(" Smart Resume Analyzer")

    uploaded_file = st.file_uploader("Upload Resume (TXT)", type=["txt"])
    job = st.text_area("Enter Job Requirements")

    if st.button("Analyze Resume"):
        if uploaded_file and job:

            resume = uploaded_file.read().decode("utf-8").lower()
            job = job.lower()

            resume_words = resume.split()
            job_words = job.split()

            matched = [w for w in job_words if w in resume_words]
            missing = [w for w in job_words if w not in resume_words]

            score = len(matched)
            percent = int((score / len(job_words)) * 100)

            st.markdown("### 📊 Analysis Dashboard")
            st.progress(percent / 100)
            st.write(f"### 🎯 Match Score: {percent}%")

            # Rating
            if percent >= 80:
                rating = "🌟 Excellent"
            elif percent >= 60:
                rating = "👍 Good"
            elif percent >= 40:
                rating = "⚠️ Average"
            else:
                rating = "❌ Poor"

            st.write(f"### ⭐ Rating: {rating}")

            # Skills display
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✅ Matched Skills")
                for skill in matched:
                    st.markdown(f"- 🟢 {skill}")

            with col2:
                st.markdown("### ❗ Missing Skills")
                for skill in missing:
                    st.markdown(f"- 🔴 {skill}")

            # Resume Preview
            st.markdown("### 📄 Resume Preview")
            st.text(resume[:300])

            # Smart suggestion
            if percent >= 70:
                st.success("🎉 Candidate Selected")
                st.info("💡 Highly suitable for the job")
            elif percent >= 40:
                st.warning("⚠️ Candidate needs improvement")
                st.info(f"💡 Improve skills: {missing}")
            else:
                st.error("❌ Candidate Rejected")

        else:
            st.warning("Upload resume and enter job")
