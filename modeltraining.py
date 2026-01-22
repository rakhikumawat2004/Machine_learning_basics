import numpy as np 
import pandas as pd

df=pd.read_csv("titanic - titanic.csv")
print(df.head(2))

df=df.drop(columns=['PassengerId','Name','Sex','Cabin','Embarked','Ticket'])
print(df.dtypes)
x=df.drop(columns=['Survived'])
y=df['Survived']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

rf.fit(x_train,y_train)

import joblib
joblib.dump(rf,'rf_model.pkl')
print("model saved as rf_model.pkl")