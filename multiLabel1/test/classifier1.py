import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_selection import chi2
from sklearn.linear_model import SGDClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from wordcloud import WordCloud
from skmultilearn.problem_transform import ClassifierChain


df = pd.read_csv("essays3.csv", encoding = "cp1252")

df_personalities = df.drop(['#AUTHID','text'], axis=1)
counts = []
categories = list(df_personalities.columns.values)
for i in categories:
    counts.append((i, df_personalities[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_persons'])
print(df_stats)

df_stats.plot(x='category', y='number_of_persons', kind='bar', legend=False, grid=True, figsize=(8, 5))
plt.title("Number of essays per personality category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)

rowsums = df.iloc[:,2:].sum(axis=1)
x=rowsums.value_counts()
#plot
plt.figure(figsize=(8,5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple personality categories per essay")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)

#turn to lower case
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#remove specail characters
df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")

#correction the words
df['text'][:5].apply(lambda x: str(TextBlob(x).correct()))
#trainDF['text'][:5].apply(lambda x: str(TextBlob(x).correct()))

#remove word less than 3 charater
df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#remove stop words
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Lemmatization
from textblob import Word, TextBlob

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#1o most frequent words removing
freq = pd.Series(' '.join(df['text']).split()).value_counts()[:10]
freq = list(freq.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#10 least frequent words removing
freq = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]
freq = list(freq.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

categories = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']

#visualize
all_words = ' '.join([text for text in df['text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)
X_train = train.text
X_test = test.text
print(X_train.shape)
print(X_test.shape)

#empath features
from empath import Empath
lexicon = Empath()
empathFeaturesTrain_x=X_train.apply(lambda x: (lexicon.analyze(x)))
empathFeaturesValid_x=X_test.apply(lambda x:(lexicon.analyze(x)))
v = DictVectorizer(sparse=False)
X = v.fit_transform(empathFeaturesTrain_x)
Y = v.fit_transform(empathFeaturesValid_x)
y_train = train.drop(labels = ['#AUTHID','text'], axis=1)
y_test = test.drop(labels = ['#AUTHID','text'], axis=1)
print(v.get_feature_names())



NB_pipeline = Pipeline([
('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
])


# SVM_pipeline = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
#                       ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                             alpha=1e-3, random_state=42,
#                                             max_iter=5, tol=None)),
# ])

########
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
# classifier = BinaryRelevance(GaussianNB())
# train
# classifier.fit(X, y_train)
# predict
# predictions = classifier.predict(Y)
# accuracy
# print("Accuracy = ",accuracy_score(y_test,predictions))
#######
#
# for category in categories:
#     print('... Processing {}'.format(category))
#     #train the model using X_dtm & y
#     NB_pipeline.fit(X, train[category])
#     #compute the testing accuracy
#     prediction = NB_pipeline.predict(Y)
#     print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

# initialize classifier chains multi-label classifier
classifier = ClassifierChain(LogisticRegression())