import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
data=pd.read_csv('mllablastday\docs.csv')
cv=CountVectorizer()
x=data['text']
y=data['class']
x=cv.fit_transform(x)
xt,xte,yt,yte=train_test_split(x,y,test_size=0.3)
model=MultinomialNB()
model.fit(xt,yt)
pred=model.predict(xte)
print("Accuracy: ",accuracy_score(yte,pred))
print("Precision: ",precision_score(yte,pred,pos_label="Positive"))
print("Recall: ",recall_score(yte,pred,pos_label="Positive"))

