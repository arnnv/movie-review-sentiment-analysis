import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
index_word = dict((v, k) for k, v in word_index.items())

model = load_model('./models/model.keras')


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    padded_review = preprocess_text(review)
    prediction = model.predict(padded_review, verbose=0)
    score = prediction[0][0]
    sentiment = 'Positive' if score > 0.5 else 'Negative'
    return sentiment, score


st.title("Movie review sentiment analysis")
st.write("Enter a review:")

user_input = st.text_area("Movie review")

if st.button("Classify"):
    sentiment, score = predict_sentiment(user_input)

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {score}")
else:
    st.write("Please enter a review.")
