# app.py

import streamlit as st
from sentiment_analysis import predict_sentiment

# Set the title of the app
st.title("Movie Review Sentiment Analysis")

# Text input for the movie review
user_review = st.text_area("Enter a movie review:", "")

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    # Debug information
    st.write("Analyzing the review...")
    print("Review:", user_review)
    
    # Predict the sentiment of the review
    sentiment = predict_sentiment(user_review)
    
    # Display the result
    st.write("The sentiment of your review is:", sentiment)
    print("Sentiment:", sentiment)