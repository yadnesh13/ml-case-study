# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load data
data = pd.read_csv("fake_or_real_news.csv")

# Preprocess data
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)

# Split data into train and test sets
X, y = data['text'], data['fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define function to train model and make predictions
@st.cache(allow_output_mutation=True)
def train_and_predict(X_train, X_test, y_train, y_test):
    # Vectorize text data
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train classifier
    clf = LinearSVC()
    clf.fit(X_train_vectorized, y_train)
    
    # Evaluate classifier
    score = clf.score(X_test_vectorized, y_test)
    
    return clf, vectorizer, score

# Train model and get score
clf, vectorizer, score = train_and_predict(X_train, X_test, y_train, y_test)

# Streamlit UI
st.title("Fake News Detector")

# Text area for user input
user_input = st.text_area("Enter news text:", "")

# Function to predict whether the news is fake or real
def predict_news(input_text):
    # Vectorize the input text
    input_vectorized = vectorizer.transform([input_text])
    # Predict using the model
    prediction = clf.predict(input_vectorized)
    # Return prediction
    return prediction[0]

# Predict button
if st.button("Predict"):
    if user_input:
        prediction = predict_news(user_input)
        if prediction >=0.5 :
            st.error("This news is predicted to be FAKE.")
        else:
            st.success("This news is predicted to be REAL.")
    else:
        st.warning("Please enter some text.")

# Display model evaluation score
st.write("Model Evaluation Score:", score)
