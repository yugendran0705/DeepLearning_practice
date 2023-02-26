import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import joblib
from sklearn.model_selection import train_test_split

df=pd.read_excel('book.xlsx')
df=df[['Mileage','Price','Age(yrs)']]
x_train,x_test,y_train,y_test=train_test_split(df[['Mileage','Age(yrs)']],df.Price,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
joblib.dump(model,'model1_ass2')
print(model.predict(x_test))