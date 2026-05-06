import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('mllablastday\synthetic_heart_disease_data.csv')
le=LabelEncoder()
for col in data.columns:
    data[col]=le.fit_transform(data[col])
    print(col," mapping",dict(zip(le.classes_,le.transform(le.classes_))))
model=GaussianNB()
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
xt,xte,yt,yte=train_test_split(x,y,test_size=0.3)
model.fit(xt,yt)
pred=model.predict(xte)
print("Model accuracy: ",accuracy_score(yte,pred))
sample=['MiddleAged','Male','Yes','High','Moderate','High']
encoded=[]
for col in sample:
    encoded.append(le.fit_transform([col])[0])
res=model.predict([encoded])
print("Heart Disease" if res[0]==1 else "No heart disease")
