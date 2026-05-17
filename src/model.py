import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

#Data Load
df=pd.read_csv(r"C:\Titanic\data\train.csv")

# Data Cleaning
df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"]=df["Sex"].map({"male":0,"female":1})
df["Embarked"]=df["Embarked"].map({"S":0,"C":1,"Q":2})

#Features and target
x=df.drop("Survived",axis=1)
y=df["Survived"]

#Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Model train
model=LogisticRegression(max_iter=500)
model.fit(x_train,y_train)

#Prediction and Evaluation
y_pred=model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#Model save
joblib.dump(model,r"C:\Titanic\models\titanic_model.pkl")
print("\nModel saved!")

