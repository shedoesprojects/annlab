import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
data=pd.read_csv('mllablastday\docs.csv')
x=data['text']
y=data['class']
cv=CountVectorizer()
x=cv.fit_transform(x)
model=MultinomialNB()
xt,xte,yt,yte=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(xt,yt)
pred=model.predict(xte)
print("accuracy: ",accuracy_score(yte,pred))
print("precision: ",precision_score(yte,pred,pos_label="Positive"))
print("recall: ",recall_score(yte,pred,pos_label="Positive"))