import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Load word index and reverse word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model('simple_rnn_imdb.h5')

# Decode review back into words
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess input text with clipping to vocab size
def preprocessing_text(text, max_features=10000):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        idx = word_index.get(word, 2) + 3   # 2 for unknown token
        if idx >= max_features:             # clip index to vocab size
            idx = 2                         # replace with unknown
        encoded_review.append(idx)
    padded_sequence = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_sequence

# Predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocessing_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit App
import streamlit as st
st.title("IMDB MOVIE REVIEW SYSTEM")
st.write('PLEASE WRITE REVIEW TO CLASSIFY AS POSITIVE OR NEGATIVE SENTIMENT')

# User input
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    preprocessed_input = preprocessing_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]:.4f}')
else:
    st.write("Please write your review.")