"""
Name - Siddhant Bhardwaj
ISTA 331 HW7
Section Leader - 
Collaborators - Vibhor Mehta, Shivansh Singh Chauhan , Abhishek Agarwal,
"""

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import math

def get_data():
    '''
    This is the get_data function.
    '''
    mnist_data = sio.loadmat('mnist-original.mat')
    X = mnist_data['data'].T
    y = mnist_data['label'][0]
    return X,y
      
def get_train_and_test_sets(X,y):
    '''
    This is the get_train_and_test_sets function.
    '''
    training_X = X[:60000]
    training_y = y[:60000]
    testing_X = X[60000:]
    testing_y = y[60000:]
    shuffled_indices = np.random.permutation(60000)
    training_X = training_X[shuffled_indices]
    training_y = training_y[shuffled_indices]
    return training_X,testing_X,training_y,testing_y
    
def train_to_data(train_X,train_y,model):
    '''
    This is the train_to_data function.
    '''
    if model == "SGD":
        mod = SGDClassifier(max_iter = 200, tol = 0.001)
        mod.fit(train_X[:10000],train_y[:10000])
    elif model == "SVM":
        mod = SVC(kernel='poly')
        mod.fit(train_X[:10000],train_y[:10000])
    else:
        mod = LogisticRegression(multi_class = 'multinomial',solver = 'lbfgs')
        mod.fit(train_X,train_y)
    return mod   

def get_confusion_matrix(model,X,y):
    '''
    This is the get_confusion_matrix function.
    '''
    predictions = model.predict(X)
    cf = confusion_matrix(y,predictions)
    return cf    
    
def probability_matrix(cf):
    '''
    This is the probability_matrix function.
    '''
    matrix = cf.copy().astype('float')
    for i in range(len(matrix)):
       matrix[i] = matrix[i]/matrix[i].sum()
    matrix = matrix.round(3)
    return matrix
   
def plot_probability_matrices(matrix1,matrix2,matrix3):
    '''
    This is the plot_probability_matrices function.
    '''
    fig, ax = plt.subplots(1,3)
    ax[0].matshow(matrix1, cmap='binary')
    ax[0].set_title('LinearSVM\n')
    ax[1].matshow(matrix2, cmap='binary')
    ax[1].set_title('LogisticRegression\n')
    ax[2].matshow(matrix3, cmap='binary')
    ax[2].set_title('PolynomialSVM\n')
    
def main():
    '''
    This is the main function.
    '''
    X,y =  get_data()
    train_X,test_X,train_Y,test_Y = get_train_and_test_sets(X,y)
    model_sgd = train_to_data(train_X,train_Y,'SGD')
    model_svc = train_to_data(train_X,train_Y,'SVM')
    model_logistic = train_to_data(train_X,train_Y,'log')
    cf_sgd = get_confusion_matrix(model_sgd,test_X,test_Y)
    cf_svc = get_confusion_matrix(model_svc,test_X,test_Y)
    cf_logistic = get_confusion_matrix(model_logistic,test_X,test_Y)
    prob_sgd = probability_matrix(cf_sgd)
    prob_svc = probability_matrix(cf_svc)
    prob_logistic = probability_matrix(cf_logistic)
    for mod in (('LinearSVM:',prob_sgd),
                ('LogisticRegression:',prob_logistic),
                ('PolynomialSVM:',prob_svc)):
        print(*mod,sep = '\n')
    plot_probability_matrices(prob_sgd,prob_logistic,prob_svc)
    plt.show()
    
if __name__ == "__main__":
    main()
    
