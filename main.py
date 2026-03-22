#Step1:Import Libraries and Load Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Load the IMDB dataset word index
word_index=imdb.get_word_index()
# reverse the word index to get the original words
reverse_word_index={value:key for key,value in word_index.items()}

#Load the pre-trained model with RELU activation function
model=load_model('simplernn_imdb.h5')

#Step2:Helper Functions
#Function to decode the review back to text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#Function to preprocess user input review
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

###step-3:Prediction Function
def predict_sentiment(review):
    preprocesses_input=preprocess_text(review)
    prediction=model.predict(preprocesses_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'negative'
    return sentiment,prediction[0][0]


## streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")
# User input
user_review = st.text_area("Movie Review")
if st.button("Classify"):
    preprocess_input = preprocess_text(user_review)

    #Make prediction
    prediction=model.predict(preprocess_input)
    sentiment='positive' if prediction[0][0]>=0.0 else 'negative'

    #Display results
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
else:
    st.write("Please enter a movie review and click 'Classify' to see the prediction.")