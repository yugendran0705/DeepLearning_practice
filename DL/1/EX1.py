import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random


pd=pd.read_excel('Book.xlsx')
#print(pd)

sx=preprocessing.MinMaxScaler()
sy=preprocessing.MinMaxScaler()

x_scaled=sx.fit_transform(pd[['area','bedrooms']])
y_scaled=sy.fit_transform(pd[['price']])
#print(x_scaled)
#y_scaled=y_scaled.reshape(20,)
#print(y_scaled)
#print(y_scaled.reshape(y_scaled.shape[0],))
#print(np.shape(x_scaled))

def sigmoid_np(x):
    import math
    return 1 / (1 + math.exp(-x))

def batch_gradient(x,y_true,epochs,learning_rate=0.01):
    number_of_features=x.shape[1]
    w=np.ones(shape=(number_of_features))
    b=0
    total_samples=x.shape[0]

    cost_list=[]
    epoch_list=[]
    
    for i in range(epochs):        
        y_predicted = np.dot(w, x.T) + b

        w_grad = -(2/total_samples)*(x.T.dot(y_true-y_predicted))
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.mean(np.square(y_true-y_predicted)) # MSE (Mean Squared Error)
        
        if i%10==0:
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list
    
def stochastic_gradient(x,y_true,epochs,learning_rate=0.01):
    number_of_features=x.shape[1]
    w=np.ones(shape=(number_of_features))
    b=0
    total_samples=x.shape[0]

    cost_list=[]
    epoch_list=[]
       
    for i in range(epochs):    
        random_index = random.randint(0,total_samples-1) # random index from total samples
        sample_x = x[random_index]
        sample_y = y_true[random_index]
        
        y_predicted = np.dot(w, sample_x.T) + b
    
        w_grad = -(2/total_samples)*(sample_x.T.dot(sample_y-y_predicted))
        b_grad = -(2/total_samples)*(sample_y-y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.square(sample_y-y_predicted)
        
        if i%100==0: # at every 100th iteration record the cost and epoch value
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list

def minibatch_gradient(X, y_true, epochs = 100, batch_size = 5, learning_rate = 0.01):
    number_of_features = X.shape[1]
    # numpy array with 1 row and columns equal to number of features. In 
    # our case number_of_features = 3 (area, bedroom and age)
    w = np.ones(shape=(number_of_features)) 
    b = 0
    total_samples = X.shape[0] # number of rows in X
    
    if batch_size > total_samples: # In this case mini batch becomes same as batch gradient descent
        batch_size = total_samples
        
    cost_list = []
    epoch_list = []
    
    num_batches = int(total_samples/batch_size)
    
    for i in range(epochs):    
        random_indices = np.random.permutation(total_samples)
        X_tmp = X[random_indices]
        y_tmp = y_true[random_indices]
        
        for j in range(0,total_samples,batch_size):
            Xj = X_tmp[j:j+batch_size]
            yj = y_tmp[j:j+batch_size]
            y_predicted = np.dot(w, Xj.T) + b
            
            w_grad = -(2/len(Xj))*(Xj.T.dot(yj-y_predicted))
            b_grad = -(2/len(Xj))*np.sum(yj-y_predicted)
            
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad
                
            cost = np.mean(np.square(yj-y_predicted)) # MSE (Mean Squared Error)
        
        if i%10==0:
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list

w,b,cost,cost_list,epoch_list=minibatch_gradient(x_scaled,y_scaled.reshape(y_scaled.shape[0],),10000)

plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list,cost_list)
plt.show()

