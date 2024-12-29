

import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb


# Load the saved model and vectorizer
with open("model1.pkl", "rb") as model_file:
    model1 = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("Quora Sentiment Analysis Model")
st.write("This app uses a machine learning model to classify if two questions are duplicates.")

# Input fields for the two questions
q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    # Combine input questions for vectorization
    questions = [q1, q2]
    # Transform using the loaded vectorizer
    query = vectorizer.transform(questions).toarray().reshape(1, -1)


    # Predict using the model
    result = model1.predict(query)[0]

    # Display the prediction result
    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')
