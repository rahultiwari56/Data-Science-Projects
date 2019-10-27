import pandas as pd
import numpy as np
#importing data
data = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                  names=["label", "message"])

#data(dt) cleaning and preprocessing
import re #this library is for regular expression
import nltk
nltk.download('stopwords')  #download nltk package 'stopwords'

from nltk.corpus import stopwords #will use to remove stopwords like ? , . ! example(line 18)
from nltk.stem.porter import PorterStemmer #for stemming purpose
ps = PorterStemmer()

corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['message'][i]) #all stopwords will be replaced by space
    review = review.lower() #it will lower all the words
    review = review.split() #splitting all sentences into words 
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#creating bag of words model   
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000) #it will take only 5000 most frequent words
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(data['label'])
y = y.iloc[:,1].values

#spliting data into training data and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
 
#training model using naive bayes algo
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(x_train, y_train)

#predicting the model
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
conf_metric = confusion_matrix(y_test,y_pred)

#to find accuracy of trained model
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred,y_test)

print("accuracy of model:",score)