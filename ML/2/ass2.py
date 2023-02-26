import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import joblib


df=pd.read_excel('book.xlsx')
'''print(df)
plt.scatter(df.Mileage,df.Price,color='red',marker='+')
plt.savefig('ex1.png')
plt.show()'''

dummies=pd.get_dummies(df['Car Model'])
merged=pd.concat([df,dummies],axis='columns')
final=merged.drop(['Car Model','Mercedez Benz C class'],axis='columns')


model=linear_model.LinearRegression()
x=final[['Mileage','Age(yrs)','Audi A5','BMW X5']]
y=final.Price
model.fit(x,y)
joblib.dump(model,'model_ass2')
print(model.predict([[59000,5,0,0]]))
print(model.score(x,y))