# AI Mini Project - Fake News Detection + Resume Screening

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_fake_news_model():
    data = {
        "text": [
            "Government passes new law",
            "Aliens landed in India",
            "Economy is improving",
            "Fake cure for diseases",
            "New technology launched by company",
            "Shocking rumor spreads online"
        ],
        "label": [1, 0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])

    model = MultinomialNB()
    model.fit(X, df["label"])

    return model, vectorizer


def fake_news_detection(model, vectorizer):
    print("\n--- Fake News Detection ---")
    news = input("Enter news text: ")

    news_vec = vectorizer.transform([news])
    pred = model.predict(news_vec)

    if pred[0] == 1:
        print("Result: Real News")
    else:
        print("Result: Fake News")


def resume_screening():
    print("\n--- Resume Screening ---")

    resume = input("Enter resume skills: ").lower()
    job = input("Enter job requirements: ").lower()

    resume_words = resume.split()
    job_words = job.split()

    score = sum(1 for word in job_words if word in resume_words)

    print(f"Matching Score: {score}")

    if score >= len(job_words)//2:
        print("Result: Candidate Selected")
    else:
        print("Result: Candidate Rejected")


def main():
    model, vectorizer = train_fake_news_model()

    while True:
        print("\n====== AI MINI PROJECT ======")
        print("1. Fake News Detection")
        print("2. Resume Screening")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            fake_news_detection(model, vectorizer)

        elif choice == "2":
            resume_screening()

        elif choice == "3":
            print("Exiting... Thank you!")
            break

        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()
