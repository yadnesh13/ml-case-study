import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer

# Load the model
nb = load('naive_bayes_model.joblib')

# Load the CountVectorizer
cv = load('count_vectorizer.joblib') # Assuming you've saved your CountVectorizer

def preprocess_text(text):
    # This function should preprocess the text in the same way as your training data
    # For example, tokenizing, removing stopwords, stemming, and combining title, subject, and text
    # This is a placeholder function
    return text

st.title('Fake News Detector')

# Create a text input widget for the article text
article_text = st.text_area("Enter the text of the article here:")

if st.button('Predict'):
    # Preprocess the input text
    processed_text = preprocess_text(article_text)
    
    # Transform the text into the format expected by the model
    X_test = cv.transform([processed_text])
    
    # Make a prediction
    prediction = nb.predict(X_test)
    
    # Display the prediction
    if prediction[0] == 1:
        st.write("The article is likely TRUE.")
    else:
        st.write("The article is likely FAKE.")
