from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris=load_iris()

#['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
#df['flower_name']=df.target.apply(lambda x: iris.target_names[x])

x_train,x_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),df.target,test_size=0.2)
model=RandomForestClassifier(n_estimators=40)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
y_predicted=model.predict(x_test)
cm=confusion_matrix(y_test,y_predicted)
print(cm)