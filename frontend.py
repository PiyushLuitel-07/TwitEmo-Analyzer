# import streamlit as st
# import pickle as pl
# import numpy as np
# import pandas as pd
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import nltk

# def stemming(content):

#   stemmed_content=re.sub('[^a-zA-Z]',' ',content)
#   stemmed_content=stemmed_content.lower()
#   stemmed_content=stemmed_content.split()
#   stemmed_content=[PorterStemmer().stem(word) for word in stemmed_content if not word in stopwords.words('english')]

#   return stemmed_content

# def main():
#     st.title("Twitter Sentiment Analysis")
#     input_text = st.text_input("Enter your tweet here:")
#     input_text = stemming(input_text)
#     #converting the textual data into into numerical data
#     vectorizer=TfidfVectorizer()
#     input_text=vectorizer.transform(input_text)
#     # Load the trained model
#     trained_model = pl.load(open("trained_model.sav", "rb"))    
#     prediction = trained_model.predict(input_text)
#     if prediction[0] == 1:
#         st.write("Tweet has Positive Sentiment")
#     else:
#         st.write("Tweet has Negative Sentiment")
    
#     # Rest of your code goes here
    
# if __name__ == "__main__":
#     main()



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
nltk.download('stopwords')

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

    # Load the trained vectorizer and model
    with open("trained_vectorizer.pkl", "rb") as f:
        vectorizer = pl.load(f)
    with open("trained_model.sav", "rb") as f:
        trained_model = pl.load(f)

    # Transform input text using the fitted vectorizer
    input_text = vectorizer.transform([input_text])

    # Make prediction
    prediction = trained_model.predict(input_text)
    if prediction[0] == 1:
        st.write("Tweet has Positive Sentiment")
    else:
        st.write("Tweet has Negative Sentiment")

if __name__ == "__main__":
    main()
