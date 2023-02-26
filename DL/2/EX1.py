import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('customer.csv')
df=df.drop(['customerID'],axis='columns')
#print(df.dtypes)
df=df[df.TotalCharges!=' ']
df.TotalCharges=pd.to_numeric(df.TotalCharges)
df=df[df.MonthlyCharges!=' ']
df.MonthlyCharges=pd.to_numeric(df.MonthlyCharges)
#print(df.shape)
'''tenure_churn_no=df[df.Churn=='No'].tenure
tenure_churn_yes=df[df.Churn=='Yes'].tenure
plt.xlabel('tenure')
plt.ylabel('Number of customers')
plt.title('Customer Churn Prediction Visualization')
plt.hist([tenure_churn_yes,tenure_churn_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()'''

'''for column in df:
    if df[column].dtype==object:
        print(f'{column}: {df[column].unique()}')'''

df=df.replace('No internet service','No')
df=df.replace('No phone service','No')
columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for column in columns:
    df[column].replace({'Yes':1,'No':0},inplace=True)
df['gender'].replace({'Male':1,'Female':0},inplace=True)
df=pd.get_dummies(data=df,columns=['InternetService','Contract','PaymentMethod'])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
columns_to_scale=['tenure','MonthlyCharges','TotalCharges']
df[columns_to_scale]=scaler.fit_transform(df[columns_to_scale])

x=df.drop('Churn',axis='columns')
x=np.asarray(x).astype(np.float32)

y=df['Churn']
y=np.asarray(y).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(26,input_shape=(26,),activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)
model.evaluate(x_test,y_test)
y_p=[]
yp=model.predict(x_test)
for element in yp:
    if element>0.5:
        y_p.append(1)
    else:
        y_p.append(0)
y_p=np.asarray(y_p)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_p))

import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_p)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
