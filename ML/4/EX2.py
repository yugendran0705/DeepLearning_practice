from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.drop(['petal length (cm)','petal width (cm)'],axis='columns',inplace=True)

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['sepal length (cm)','sepal width (cm)']])
df['cluster']=y_predicted

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='green')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='red')
plt.scatter(df3['sepal length (cm)'],df3['sepal width (cm)'],color='black')
plt.plot(np.average(df1['sepal length (cm)']),np.average(df1['sepal width (cm)']),'o',color='blue')
plt.plot(np.average(df2['sepal length (cm)']),np.average(df2['sepal width (cm)']),'o',color='blue')
plt.plot(np.average(df3['sepal length (cm)']),np.average(df3['sepal width (cm)']),'o',color='blue')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
#plt.show()

sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['sepal length (cm)','sepal width (cm)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
#plt.show()

X_train,X_test,y_train,y_test=train_test_split(df[['sepal length (cm)','sepal width (cm)']],iris.target,test_size=0.2)
print(cross_val_score(RandomForestClassifier(n_estimators=40),X_train,y_train))
print(np.average(cross_val_score(RandomForestClassifier(n_estimators=40),X_train,y_train)))
print(cross_val_score(SVC(),X_train,y_train))
print(np.average(cross_val_score(SVC(),X_train,y_train)))