# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:53:44 2017

@author: erik
"""

import numpy as np
from matplotlib import pyplot as plt

def noise(minimum,maximum):
    return (maximum - minimum) * np.random.random_sample() + minimum

def generate_data(n):
    
    data = np.ndarray((n,2))
    data[:,0] = np.random.uniform(size=n)#np.linspace(0.0,1.0,n) 
    data[:,1] = map(lambda x: np.sin(2*np.pi*x) + noise(-0.1, 0.1), data[:,0])
    
    return data

def generate_test_data(n):
    
    data = np.ndarray((n,2))
    data[:,0] = np.linspace(0.0,1.0,n) 
    data[:,1] = map(lambda x: np.sin(2*np.pi*x) + noise(-0.1, 0.1), data[:,0])
    
    return data

def generate_noiseless_data(n):
    
    data = np.ndarray((n,2))
    data[:,0] = np.linspace(0.0,1.0,n) 
    data[:,1] = map(lambda x: np.sin(2*np.pi*x) , data[:,0])
    
    return data

def polynomial(x,theta,degree):
    hypothesis = 0
    #
    new_x = list()
    for j in range(degree+1):
        hypothesis += theta[j] * np.power(x ,j)
        #
        new_x.append(np.power(x,j))
    return hypothesis,np.array(new_x)
    
    
def sgd(training_data,alpha,degree,iterations):
    np.random.seed(0)
    theta = np.random.rand(1,degree+1)[0]
    for it in range(iterations):
        for sample in training_data:
            x = sample[0]             
            y = sample[1]
            hypothesis,x = polynomial(x,theta,degree)
            error = y - hypothesis
           # print error.shape
            #print x.shape
            theta = theta + alpha * error * x
   #         regression = map(lambda x: polynomial(x,theta,degree)[0],training_data[:,0])
  #      print mse(regression, training_data[:,1])
    return theta
    
def mse(regression, target):
    error = regression - target
    squared_error = error**2
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error
    
    
if __name__  == '__main__':
    training_data = generate_data(100)
    alpha = 0.2
    degree = 15
    iterations = 1000
    
    theta = sgd(training_data[:,:],alpha,degree,iterations)
    test_data = generate_test_data(1000)
    noise_less_data = generate_noiseless_data(1000)
    regression = map(lambda x: polynomial(x,theta,degree)[0],test_data[:,0])
    
    print theta, mse(regression, test_data[:,1])
    
    plt.figure(0)
    plt.plot(training_data[:,0],training_data[:,1],'bx')
    plt.plot(test_data[:,0],regression,'g-')
    plt.plot(noise_less_data[:,0],noise_less_data[:,1],'r-')
    plt.show()
