import streamlit as st
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection App")

# -----------------------------
# Load and Train Model
# -----------------------------
@st.cache_resource
def train_model():
    # Load LOCAL dataset (important for deployment)
    fake = pd.read_csv("Fake_small.csv")
    real = pd.read_csv("True_small.csv")

    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)

    # Text cleaning
    def clean_text(text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Accuracy
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    return model, vectorizer, acc, cm, data


model, vectorizer, accuracy, cm, data = train_model()

# -----------------------------
# Show Accuracy
# -----------------------------
st.write(f"### Model Accuracy: {round(accuracy, 2)}")

# -----------------------------
# Graph: Data Distribution
# -----------------------------
st.subheader("📊 Data Distribution")

fig, ax = plt.subplots()
sns.countplot(x="label", data=data, ax=ax)
ax.set_title("Label (0 = Fake, 1 = Real)")
st.pyplot(fig)

# -----------------------------
# Word Cloud
# -----------------------------
st.subheader("☁ Word Cloud (Fake News)")

fake_text = " ".join(data[data["label"] == 0]["text"].values)

wordcloud = WordCloud(width=800, height=400, background_color="black").generate(fake_text)

fig_wc, ax_wc = plt.subplots()
ax_wc.imshow(wordcloud, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)

# -----------------------------
# User Input
# -----------------------------
st.subheader("📝 Enter News Text:")

user_input = st.text_area("")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        input_vector = vectorizer.transform([user_input])

        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0]

        if prediction == 1:
            st.success(f"✅ Real News (Confidence: {round(probability[1]*100, 2)}%)")
        else:
            st.error(f"❌ Fake News (Confidence: {round(probability[0]*100, 2)}%)")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📉 Confusion Matrix")

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)
