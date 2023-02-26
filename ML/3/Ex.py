import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('HR_comma_sep.csv')

#pd.crosstab(df.salary,df.left).plot(kind='bar') 
#plt.show()
 
df1=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary','left']]
dummies=pd.get_dummies(df.salary)
merged=pd.concat([df,dummies],axis='columns')
final=merged.drop(['salary','low'],axis='columns')
x=final[['satisfaction_level','average_montly_hours','promotion_last_5years','high','medium']]
y=final.left
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print(model.predict(x_test))
print(model.score(x_test,y_test))


