# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hR6TO3AtubDmRoiPwTETH65VdBErDUDR
"""

! pip install kaggle

# prompt: write a code to configure path of kaggle.json file

!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d kazanova/sentiment140

from zipfile import ZipFile
dataset='/content/sentiment140.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

twitter_data=pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')

twitter_data.shape

twitter_data.head(1)

#naming the column and reading the dataser again
column_names=['target','id','date','flag','user','text']
twitter_data=pd.read_csv('/content/training.1600000.processed.noemoticon.csv',names=column_names,encoding='ISO-8859-1')

twitter_data.head()

twitter_data.isnull().sum()

twitter_data['target'].value_counts()

#convert the target 4 to 1
twitter_data.replace({'target':{4:1}},inplace=True)

"""**Steming**"""

twitter_data['target'].value_counts()

port_stem=PorterStemmer()

def stemming(content):

  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)

  return stemmed_content

twitter_data['stemmed_content']=twitter_data['text'].apply(stemming)



#separating data and label
X=twitter_data['stemmed_content'].values
Y=twitter_data['target'].values

"""Splitting the data into training data and test data respectively"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

print(Y.shape,Y_train.shape,Y_test.shape)

#converting the textual data into into numerical data

vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train)
X_Test=vectorizer.transform(X_test)


#Training the machine learning model   using Logistic Regression

model=LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)

#accuracy score on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy score on training data :',training_data_accuracy)

#accuracy score on test data

X_test_prediction=model.predict(X_Test)

test_data_accuracy=accuracy_score(Y_test,X_test_prediction)

print('Accuracy score on training data :',test_data_accuracy)

import pickle

filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))

