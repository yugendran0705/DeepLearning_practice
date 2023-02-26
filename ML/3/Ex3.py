import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

digits = load_digits()

#['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']
#print(dir(digits))

df=pd.DataFrame(digits.data,digits.target)
df['target']=digits.target

model1=SVC(kernel='rbf')
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'), df.target, test_size=0.2)
model1.fit(X_train, y_train)
print(model1.score(X_test, y_test))

model2=SVC(kernel='linear')
model2.fit(X_train, y_train)
print(model2.score(X_test, y_test))


