# Streamlit app for fake news detection

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# load the model and vectorizer and data
model=joblib.load('../Outputs/models/logistic_regression_model.pkl')
vectorizer=joblib.load('../Outputs/models/tfidf_vectorizer.pkl')
data=pd.read_csv('../data/cleaned_data.csv')
data['Label'] = data['Label'].map({0: 'Real', 1: 'Fake'})
data['Label'] = data['Label'].astype('category')

# input for user to enter text
st.title("Fake News Detection")
st.write("Enter the text to be classified as Fake or Real News")
user_input = st.text_area("Enter the text here", height=200)
st.write("Click on the button to classify the text")
stopword = set(stopwords.words('english'))
stemmer = PorterStemmer()
if st.button("Classify"):
    # Preprocess the input text
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = [stemmer.stem(word) for word in text]
        text = " ".join(text)
        return text

    preprocessed_input = preprocess_text(user_input)
    input_vectorized = vectorizer.transform([preprocessed_input])

    prediction = model.predict(input_vectorized)[0]
    if prediction == 0:
        prediction = 'Real'
    else:
        prediction = 'Fake'
    st.write(f"The entired data is : **{prediction}**")


# User can select which graphs to view from sidebar
st.sidebar.title("Data Visualization")
st.sidebar.write("Visualize the data to understand the distribution of Fake and Real news")
# options for the user to select the graphs like countplot of sentimental analysis or wordcloud or bigrams
options=['Sentnimental Analysis', 'Count Plots', 'Word Cloud']
option = st.sidebar.selectbox("Select which plots would you like to view", options)

if option == 'Sentnimental Analysis':
    # Count of Sentiment by Label
    st.write("Below is the count of sentiment by label")
    cp_senti_label= '../Outputs/plots/Count_of_Sentiment_by_Label.png'
    img = plt.imread(cp_senti_label)
    st.image(img, caption='Count of Sentiment by Label', use_container_width=True)
    # Count of Sentiment by Subject
    st.write("Below is the count of sentiment by subject")
    cp_senti_sub= '../Outputs/plots/Count_of_Sentiment_by_Subject.png'
    img = plt.imread(cp_senti_sub)
    st.image(img, caption='Count of Sentiment by Subject', use_container_width=True)
    # Sentiment Distribution of News
    st.write("Below is the sentiment distribution of all news articles (true and fake)")
    cp_senti_dist= '../Outputs/plots/Sentiment_Distribution_of_News_Articles.png'
    img = plt.imread(cp_senti_dist)
    st.image(img, caption='Sentiment Distribution of News Articles', use_container_width=True)

    # Bigrams
    st.write("Below is the bigram of news articles content")
    cp_bigrams_text='../Outputs/plots/Top_20_Bigrams_by_TFIDF_Score.png'
    img = plt.imread(cp_bigrams_text)
    st.image(img, caption='Top 20 Bigrams by TFIDF Score', use_container_width=True)
    st.write("Below is the bigram of news articles title")
    cp_bigrams_title='../Outputs/plots/Top_20_Bigrams_by_TFIDF_Score_for_Title.png'
    img = plt.imread(cp_bigrams_title)
    st.image(img, caption='Top 20 Bigrams by TFIDF Score Title', use_container_width=True)

elif option == 'Count Plots':
    # Count Plot of True vs Fake News
    st.write("Below is the count plot of true vs fake news")
    cp_count_plot= '../Outputs/plots/Count_of_True_vs_Fake_News.png'
    img = plt.imread(cp_count_plot)
    st.image(img, caption='Count Plot of True vs Fake News', use_container_width=True)
    # Count Plot of True Vs Fake News distributed monthly
    st.write("Below is the count plot of true vs fake news distributed monthly")
    cp_count_plot_month='../Outputs/plots/Count_of_True_vs_Fake_News_by_Month.png'
    img = plt.imread(cp_count_plot_month)
    st.image(img, caption='Count of True vs Fake News by Month', use_container_width=True)
    # Count Plot of True Vs Fake News distributed by year
    st.write("Below is the count plot of true vs fake news distributed by year")
    cp_count_plot_year='../Outputs/plots/Count_of_True_vs_Fake_News_by_Year.png'
    img = plt.imread(cp_count_plot_year)
    st.image(img, caption='Count of True vs Fake News by Year', use_container_width=True)
    # Count Plot of True Vs Fake News distributed by subject
    st.write("Below is the count plot of true vs fake news distributed by subject")
    cp_count_plot_subject='../Outputs/plots/Count_of_True_vs_Fake_News_by_Subject.png'
    img = plt.imread(cp_count_plot_subject)
    st.image(img, caption='Count of True vs Fake News by Subject', use_container_width=True)

elif option == 'Word Cloud':    
    # Word Cloud of All news
    st.write("Below is the word cloud of All news (true and fake)")
    cp_wordcloud_all='../Outputs/plots/Word_Cloud_for_All_News.png'
    img = plt.imread(cp_wordcloud_all)
    st.image(img, caption='Word Cloud of All News', use_container_width=True)
    # Word Cloud of Title
    st.write("Below is the word cloud for Title")
    cp_wordcloud_title='../Outputs/plots/Word_Cloud_for_Title.png'
    img = plt.imread(cp_wordcloud_title)
    st.image(img, caption='Word Cloud of Title', use_container_width=True)

# Confusion Matrix for the Model
st.write("Below is the confusion matrix for the model")
st.write("The model achieved an accuracy of 0.99 on the test data along with a precision of 0.99, recall of 0.99 and f1 score of 0.99")
cp_confusion_matrix='../Outputs/plots/confusion_matrix.png'
img = plt.imread(cp_confusion_matrix)
st.image(img, caption='Confusion Matrix', use_container_width=True)





