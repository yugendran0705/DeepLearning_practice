import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split

df=pd.read_csv('Churn_Modelling.csv')
df=df.drop(['RowNumber','CustomerId','Surname'],axis='columns')
df=pd.get_dummies(data=df,columns=['Geography','Gender'])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']]=scaler.fit_transform(df[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']])

x=df.drop('Exited',axis='columns')
x=np.asarray(x).astype(np.float32)

y=df['Exited']
y=np.asarray(y).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

#print(x_train.shape)

import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(26,input_shape=(13,),activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)
model.evaluate(x_test,y_test)
y_p=[]
yp=model.predict(x_test)
for i in yp:
    if i>=0.5:
        y_p.append(1)
    else:
        y_p.append(0)

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_p))
cm=confusion_matrix(y_test,y_p)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
