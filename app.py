import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import nltk
nltk.download('stopwords')
nltk.download('punkt')
#---------------------------importing data and training the ocuntvectorizer----------------------------------------------
df=pd.read_csv('preprocessed_twitter_data.csv')

df = df.dropna(subset=['clean_tweet'])
df = df.reset_index(drop=True)

bow_vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=5000, stop_words='english')

bow_vectorizer.fit(df['clean_tweet'])
#----------------------------------Defining preprocessing functions-------------------------------------------
nltk.download('wordnet')

lemma = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

def cleaning_URLs(data):
    txt = re.sub(r"(https?://|www\.)\S+", ' ', data)
    return re.sub(r"(www\.|https?://)", ' ', txt)

def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct

def cleaning_numbers(txt):
    return re.sub('[0-9]+', '', txt)

def remove_stopwords(txt):
    txt_clean = " ".join([word for word in txt.split() if word not in stopwords])
    return txt_clean

def remove_repeating_characters(text):
    cleaned_text = re.sub(r'(\w)\1+', r'\1', text)
    return cleaned_text

def tokenize(txt):
    tokens = word_tokenize(txt)
    return tokens

def lemmatization(token_txt):
    text = [lemma.lemmatize(word) for word in token_txt]
    return text

model=joblib.load('BOW_model.joblib')
#-----------------------------------------------------Flask App---------------------------------------------------------------------------------------


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def process_text():
    data = request.json
    input = data.get('text')

    input=input.lower()
    #Removing URLs
    input=cleaning_URLs(input)
    #Removing punctuations
    input=remove_punctuation(input)
    #Removing numbers
    input=cleaning_numbers(input)
    #removing stopwords
    input=remove_stopwords(input)
    #Removing repeating characters
    input=remove_repeating_characters(input)
    #Tokenizing
    input=tokenize(input)
    #lemmantizing
    input=lemmatization(input)
    #joining tokens
    input=" ".join(input)

    bow_input = bow_vectorizer.transform([input])
    bow_pred = model.predict(bow_input)

    if bow_pred[0]==0:
        sentiment='Negative'
    else:
        sentiment='Positive'
    
    return jsonify({'processed_text':sentiment})

if __name__ == '__main__':
    app.run(debug=True)

