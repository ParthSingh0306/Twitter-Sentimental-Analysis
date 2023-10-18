from google.colab import drive
drive.mount('/content/drive')

# **Data Loading**

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import nltk
import seaborn as sns

#importing Dataset
DATASET_COLUMNS=['target','ids','date','flag','user','text']
dframe = pd.read_csv('/content/drive/MyDrive/Projects/Sentiment analysis/twitter_dataset.csv', encoding='ISO-8859-1',names=DATASET_COLUMNS)
dframe.head()

dframe.info()

#checking for any null comments
dframe.isnull().any()

#Identifying the targets
unique_targets = dframe['target'].unique()
print(unique_targets)

#converting 4 to 1
dframe['target'].replace(4, 1, inplace=True)

unique_targets = dframe['target'].unique()
print(unique_targets)

#checking rows and column
dframe.shape

Data Visualization

# Counting the occurrences of each unique value in the 'target' column
target_counts = dframe['target'].value_counts()
print(target_counts)

num_zeros = target_counts[0]

num_ones = target_counts[1]

# Creating the pie chart
plt.figure(figsize=(6, 6))
plt.pie([num_zeros, num_ones], labels=['Negative: 0', 'Positive: 1'], colors=['red', 'lightgreen'], autopct='%1.1f%%', startangle=90)

plt.title('Distribution of Target Values')

plt.show()

#Checking sentiment distribution over time
dframe['date'] = pd.to_datetime(dframe['date'])

unique_targets = dframe['date'].unique()
print(unique_targets)
print(unique_targets.shape)

sentiment_time = dframe.groupby([pd.Grouper(key='date', freq='D'), 'target']).size().unstack(fill_value=0)
print(sentiment_time)

sentiment_time.rename({0: 'Negative', 1: 'Positive'}, axis=1, inplace=True)

print(sentiment_time)

plt.figure(figsize=(20, 6))
plt.plot(sentiment_time.index, sentiment_time['Negative'], marker='o', label='Negative', color='red')
plt.plot(sentiment_time.index, sentiment_time['Positive'],marker='o', label='Positive', color='green')

plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Count of Positive and Negative Comments Over Time')
plt.legend()

# Set the y-axis limits to include zero
plt.ylim(bottom=0)

plt.show()


# Plotting histogram of text lengths
text_lengths = dframe['text'].apply(len)

print(text_lengths)

# Count the occurrences of each unique text length
text_length_counts = text_lengths.value_counts().sort_index()

print(text_length_counts)

plt.figure(figsize=(12, 6))
plt.hist(text_lengths, bins=20,edgecolor='black')
plt.xlabel('Text Length')
plt.ylabel('Number of Samples')
plt.title('Text Length Distribution')
plt.show()

print('Mean text Length')
print(np.mean(text_lengths))

positive_texts = text_lengths[dframe['target'] == 1]
negative_texts = text_lengths[dframe['target'] == 0]

print(type(positive_texts))

print(positive_texts)
print(negative_texts)

print('Mean Positive text Length')
print(np.mean(positive_texts))

print('Mean Negative text Length')
print(np.mean(negative_texts))

plt.figure(figsize=(12, 6))
plt.hist(positive_texts, bins=20, color='green', alpha=0.6, edgecolor='black', label='Positive')
plt.xlabel('Text Length')
plt.ylabel('Number of Samples')
plt.title('Text Length Distribution by Sentiment')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.hist(negative_texts, bins=20, color='red', alpha=0.6, edgecolor='black', label='Negative')
plt.xlabel('Text Length')
plt.ylabel('Number of Samples')
plt.title('Text Length Distribution by Sentiment')
plt.legend()
plt.show()

p1=sns.kdeplot(positive_texts, shade=True, color="r").set_title('Distribution of Number Of positive and Negative words')
p1=sns.kdeplot(negative_texts, shade=True, color="g")

plt.xlabel('Text Length')
plt.ylabel('Density')

plt.legend()
plt.show()

#Counting sentiments with respect to the users
sentiment_counts = dframe.groupby(['user', 'target']).size().unstack(fill_value=0)

sentiment_counts_positive = sentiment_counts.sort_values(by=1, ascending=False)
sentiment_counts_negative = sentiment_counts.sort_values(by=0, ascending=False)

print(sentiment_counts)

positive_counts_sorted = sentiment_counts_positive[1]
negative_counts_sorted = sentiment_counts_negative[0]

print(positive_counts_sorted)
print(negative_counts_sorted)

print(positive_counts_sorted[0:49])

print(positive_counts_sorted[positive_counts_sorted==max(positive_counts_sorted)],max(positive_counts_sorted))

print(negative_counts_sorted[negative_counts_sorted==max(negative_counts_sorted)],max(negative_counts_sorted))

plt.figure(figsize=(30,30))
plt.pie(positive_counts_sorted[0:49], labels=positive_counts_sorted.index[0:49], autopct='%1.1f%%')

plt.title('Distribution of Target Values')

plt.show()

plt.figure(figsize=(30,30))
plt.pie(negative_counts_sorted[0:49], labels=negative_counts_sorted.index[0:49], autopct='%1.1f%%')

plt.title('Distribution of Target Values')

plt.show()

# **Data Preprocessing**

df = dframe[['target', 'text']]
df.head()

#Converting every uppercase to lowercase
df['clean_text']=df['text'].str.lower()
df.head()

#Removing URLs
data="@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it"
txt=re.sub(r"(https?://|www\.)\S+",' ',data)#removing the entire url
txt=re.sub(r"(www\.|https?://)",' ',txt)#removing broken url(incomplete)
print(txt)

def cleaning_URLs(data):
    txt=re.sub(r"(https?://|www\.)\S+",' ',data)
    return re.sub(r"(www\.|https?://)",' ',txt)

df['clean_text'] = df['clean_text'].apply(lambda x: cleaning_URLs(x))
df.head()

#Removing punctuations
print(string.punctuation)

txt="@switchfoot - awww, that's a bummer"
txt_nopunct="".join([char for char in txt if char not in string.punctuation]) #character comparasion is done not word comparasion
print(txt_nopunct)

def remove_punctuation(txt):
 txt_nopunct="".join([c for c in txt if c not in string.punctuation])
 return txt_nopunct

df['clean_text']=df['clean_text'].apply(lambda x:remove_punctuation(x))
df.head()

#Removing numbers
txt="1234  how are you 456"
txt_withoutnum=re.sub('[0-9]+', '',txt)
print(txt_withoutnum)

def cleaning_numbers(txt):
    return re.sub('[0-9]+', '', txt)

df['clean_text'] = df['clean_text'].apply(lambda x: cleaning_numbers(x))
df.head()

nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words('english')
print(stopwords)

txt="is upset that he cant update his facebook by"
txt_clean=" ".join([word for word in txt.split() if word not in stopwords])
print(txt_clean)

def remove_stopwords(txt):
  txt_clean=" ".join([word for word in txt.split() if word not in stopwords])
  return txt_clean

df['clean_text']=df['clean_text'].apply(lambda x:remove_stopwords(x))
df.head()

#Removing repeating characters
#Repeating characters are characters that occur consecutively multiple times in a sequence. For example, in the word "hello," the letter 'l' is repeated twice consecutively,
text="Helloooooooo success"
cleaned_text = re.sub(r'(\w)\1+', r'\1', text)
print(cleaned_text)

def remove_repeating_characters(text):
    cleaned_text = re.sub(r'(\w)\1+', r'\1', text)
    return cleaned_text

df['clean_text']=df['clean_text'].apply(lambda x:remove_repeating_characters(x))
df.head()

#Tokenizing
def tokenize(txt):
  tokens=re.split('\W+',txt)
  return tokens

df['clean_text']=df['clean_text'].apply(lambda x:tokenize(x))
df.head()

#When the code is executed, it will split the txt into a list of tokens using the regular expression \W+.
#This means that any non-word characters (such as punctuation marks, spaces, etc.)
#will act as separators, and the text will be divided into individual words or tokens.



nltk.download('wordnet')
nltk.download('omw-1.4')
lemma= nltk.WordNetLemmatizer()

def lemmatization(token_txt):
  text=[lemma.lemmatize(word) for word in token_txt]
  return text
df['clean_text']=df['clean_text'].apply(lambda x:lemmatization(x))
df.head()

#lemmantizing
token_txt="The cats are chasing mice and playing with balls."
text=[lemma.lemmatize(word) for word in token_txt.split()]
print(text)

df.head()

range(len(df['clean_text'])-1)

df['clean_text']

tokenized_tweet=[]
for i in range(len(df['clean_text'])):
    tokenized_tweet.append(" ".join(df['clean_text'][i]))

df['clean_tweet'] = tokenized_tweet
df.head()

df.to_csv('preprocessed_twitter_data.csv', index=False)

all_words = " ".join([sentence for sentence in df['clean_tweet'][0:100]])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

all_words = " ".join([sentence for sentence in df['clean_tweet'][df['target'] == 1][0:100]])


wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

all_words = " ".join([sentence for sentence in df['clean_tweet'][df['target'] == 0][0:100]])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

df['clean_tweet'][df['target'] == 0][0:99]

# **Model Training**

!pip install joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
words=[100,500,1000,2000,5000,7000,10000,20000,50000,100000,200000,500000,2000000,5000000,7000000,10000000,12000000,14000000,17000000,20000000]

i=0
precisions=[]
for size in words:
  bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=size, stop_words='english')
  bow = bow_vectorizer.fit_transform(df['clean_tweet'])
  x_train, x_test, y_train, y_test = train_test_split(bow, df['target'], random_state=42, test_size=0.30)
  model = LogisticRegression()
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  precison=precision_score(y_test, pred)
  precisions.append(precison)
  print(i)
  i=i+1

plt.figure(figsize=(20, 6))
plt.plot(words,precisions, marker='o',color='red')

plt.xlabel('words count')
plt.ylabel('precision')
plt.title('Count of Precison against word count')

plt.legend()

# Setting the y-axis limits to include zero
plt.ylim(bottom=0,top=1)

plt.show()

plt.figure(figsize=(20, 6))
plt.plot(index,precisions, marker='o', color='red')

plt.xlabel('words count')
plt.ylabel('precision')
plt.title('Count of Precison against word count')

plt.xticks(index,index)
plt.legend()

# Set the y-axis limits to include zero
plt.ylim(bottom=0,top=1)

plt.show()

i=0
precisions=[]
for size in words:
  bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=size, stop_words='english')
  bow = bow_vectorizer.fit_transform(df['clean_tweet'])
  x_train, x_test, y_train, y_test = train_test_split(bow, df['target'], random_state=42, test_size=0.30)
  model=MultinomialNB()
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  precison=precision_score(y_test, pred)
  precisions.append(precison)
  print(i)
  i=i+1

plt.figure(figsize=(20, 6))
plt.plot(words,precisions, marker='o',color='red')

plt.xlabel('words count')
plt.ylabel('precision')
plt.title('Count of Precison against word count')

plt.legend()

# Set the y-axis limits to include zero
plt.ylim(bottom=0,top=1)

plt.show()

plt.figure(figsize=(20, 6))
plt.plot(index,precisions, marker='o', color='red')

plt.xlabel('words count')
plt.ylabel('precision')
plt.title('Count of Precison against word count')

plt.xticks(index,index)
plt.legend()

# Set the y-axis limits to include zero
plt.ylim(bottom=0,top=1)

plt.show()

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=5000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

x_train, x_test, y_train, y_test = train_test_split(bow, df['target'], random_state=42, test_size=0.30)

model = LogisticRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)
precison=precision_score(y_test, pred)

print(precison)

model_filename = 'sentiment_logistic_model.joblib'
joblib.dump(model, model_filename)

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=5000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

x_train, x_test, y_train, y_test = train_test_split(bow, df['target'], random_state=42, test_size=0.30)

model=MultinomialNB()
model.fit(x_train, y_train)

pred = model.predict(x_test)
precison=precision_score(y_test, pred)

print(precison)

model_filename = 'sentiment_NaiveBayes_model.joblib'
joblib.dump(model, model_filename)

logistic_model = joblib.load('/content/drive/MyDrive/Projects/Sentiment analysis/sentiment_logistic_model.joblib')
NaiveBayes_model = joblib.load('/content/drive/MyDrive/Projects/Sentiment analysis/sentiment_NaiveBayes_model.joblib')

bow_ensemble_model_hard_1 = VotingClassifier(estimators=[('logistic_model', logistic_model), ('NaiveBayes_model', NaiveBayes_model)], voting='hard')

bow_ensemble_model_hard_1.fit(x_train, y_train)

ensemble_pred = bow_ensemble_model_hard_1.predict(x_test)


precison=precision_score(y_test,ensemble_pred)

print(precison)

model_filename = 'BOW_model.joblib'
joblib.dump(model, model_filename)






