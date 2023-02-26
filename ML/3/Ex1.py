from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()

# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']
#print(iris.data[0])
#print(iris.target)



model=LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
model.fit(X_train, y_train)
m=model.predict([[5.1, 3.5, 1.4, 0.2]])
for i in m:
    if(i==0):
        print('setosa')
    elif(i==1):
        print('versicolor')
    else:
        print('virginica')

print(model.score(X_test, y_test))

#confusion matrix
from sklearn.metrics import confusion_matrix
y_predicted=model.predict(X_test)
cm=confusion_matrix(y_test, y_predicted)
print(cm)
