import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import joblib

df=pd.read_excel('book1.xlsx')

reg=linear_model.LinearRegression()
reg.fit(df[['year']],df['per capita income (US$)'])

print(reg.predict([[2020]]))
'''plt.scatter(df.year,df['per capita income (US$)'],color='red',marker='+')
plt.plot(df.year,reg.predict(df[['year']]),color='blue')
plt.savefig('ex1.png')
plt.show()'''

joblib.dump(reg,'model_ex1')
model=joblib.load('model_ex1')
print(model.predict([[2020]]))