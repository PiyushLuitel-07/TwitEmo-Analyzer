import streamlit as st
import pickle as pl
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure nltk resources are downloaded
# nltk.download('stopwords')

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [PorterStemmer().stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

def main():
    st.title("Twitter Sentiment Analysis")
    input_text = st.text_input("Enter your tweet here:")
    input_text = stemming(input_text)

    # Load the trained model
    with open("trained_model.sav", "rb") as f:
        trained_model = pl.load(f)

    # Load the TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pl.load(f)

    # Transform input text using the vectorizer
    input_text = vectorizer.transform([input_text])

    # Make prediction
    prediction = trained_model.predict(input_text)
    print(prediction)

    if prediction[0] == 1:
      st.write("Tweet has Positive Sentiment")
    else:
      st.write("Tweet has Negative Sentiment")

if __name__ == "__main__":
    main()
