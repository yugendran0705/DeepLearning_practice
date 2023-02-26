import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_excel('train.xlsx')
df=df[['Survived','Pclass','Sex','Age','Fare']]

df.Age=df.Age.fillna(df.Age.mean())
df.Fare=df.Fare.fillna(df.Fare.mean())

sex_le=LabelEncoder()
df['Sex_n']=sex_le.fit_transform(df['Sex'])
df=df.drop(['Sex'],axis='columns')
#print(df[0:5])
#['Survived', 'Pclass', 'Age', 'Fare]

x=df[['Pclass','Age','Fare','Sex_n']]
y=df.Survived
model=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
model.fit(x_train,y_train)
print(model.predict([[3,22.0,7.25,1]]))
print(model.score(x_test,y_test))
