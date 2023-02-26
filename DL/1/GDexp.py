import numpy as np

def log_loss(y_true,y_predicted):
    eppsilon=1e-15
    y_predicted_new=[max(i,eppsilon) for i in y_predicted]
    y_predicted_new=[min(i,1-eppsilon) for i in y_predicted_new]
    y_predicted_new=np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

def sigmoid(x):
        import math
        return 1 / (1 + math.exp(-x))

def gradient_descent(age,affordability,y_true,epochs,loss_threshold):
    w1=w2=1
    bias=0
    rate=0.5
    n=len(age)
    for i in range(epochs):
        weighted_sum=w1*age+w2*affordability+bias
        y_predicted=sigmoid(weighted_sum)
        loss=log_loss(y_true,y_predicted)
        w1d=(1/n)*np.dot(np.transpose(age),(y_predicted-y_true))
        w2d=(1/n)*np.dot(np.transpose(affordability),(y_predicted-y_true))
        bias_d=np.mean(y_predicted-y_true)
        w1=w1-rate*w1d
        w2=w2-rate*w2d
        bias=bias-rate*bias_d
        print(f'Epoch:{i},w1:{w1},w2:{w2},bias:{bias},loss:{loss}')
        if loss<=loss_threshold:
            break
    return w1,w2,bias

