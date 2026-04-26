import pandas as pd
import numpy as np
import string
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# Load and Train Model (Cached)
# -------------------------------
@st.cache_resource
def train_model():
    # Load datasets
    #fake = pd.read_csv("Fake.csv")
    ##real = pd.read_csv("True.csv")
    fake = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_news/Fake.csv")
    real = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_news/True.csv")

    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)

    # Clean text
    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    data["text"] = data["text"].apply(clean_text)

    # Split
    x = data["text"]
    y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # Model
    model = LogisticRegression()
    model.fit(x_train_vec, y_train)

    # Predictions
    pred = model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    return model, vectorizer, accuracy, cm,data


# Load trained model
model, vectorizer, accuracy, cm,data = train_model()


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 Fake News Detection using NLP & Machine Learning")

st.write(f"Model Accuracy: {accuracy:.2f}")


# -------------------------------
# Confusion Matrix Display
# -------------------------------
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

st.pyplot(fig)

st.subheader("Fake vs Real News Distribution")

fig2, ax2 = plt.subplots()
data['label'].value_counts().plot(kind='bar', ax=ax2)

ax2.set_xlabel("Label (0 = Fake, 1 = Real)")
ax2.set_ylabel("Count")
ax2.set_title("Distribution of Fake and Real News")

st.pyplot(fig2)

st.subheader("Word Cloud (Fake News)")

fake_text = " ".join(data[data['label'] == 0]['text'])

wc = WordCloud(width=800, height=400, background_color='black').generate(fake_text)

fig3, ax3 = plt.subplots()
ax3.imshow(wc)
ax3.axis("off")

st.pyplot(fig3)
# -------------------------------
# Prediction Section
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


user_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)

    if result[0] == 0:
        st.error("❌ Fake News")
    else:
        st.success("✅ Real News")
