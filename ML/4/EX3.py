from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd

wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df['target']=wine.target
scaler=MinMaxScaler()
scaler.fit(df.drop('target',axis=1))

x_train,x_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.3,random_state=100)
model=GaussianNB()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
model2=MultinomialNB()
model2.fit(x_train,y_train)
print(model2.score(x_test,y_test))
