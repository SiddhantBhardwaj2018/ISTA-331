"""
Name - Siddhant Bhardwaj
ISTA 331 Special Assignment - Penalized Linear Models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def get_frames():
    '''
    This function reads the csv file Boston_Housing.csv and sets a seed using the numpy library
    and then obtains the training set, and the testing set.
    '''
    df = pd.read_csv('Boston_Housing.csv')
    np.random.seed(95)
    training = np.random.choice(df.index,100,replace = False)
    train_df = df.iloc[training]
    test_df = df.drop(training)
    train_X = train_df.drop(['MEDV'],axis = 1)
    train_y = train_df['MEDV']
    test_X = test_df.drop(['MEDV'],axis = 1)
    test_y = test_df['MEDV']
    return train_X,train_y,test_X,test_y
    
def best_ridge(train_X,train_y,alpha_lst):
    '''
    This function performs cross-validation and then returns the 
    alpha which produces the lowest validation error.
    '''
    val_errors = []
    for i in alpha_lst:
        ridge = Ridge(alpha = i)
        score = cross_val_score(ridge,train_X,train_y,scoring = 'neg_mean_squared_error',cv = 5)
        errors = np.sum(-score)
        val_errors.append(np.sqrt(errors))
    return alpha_lst[np.argmin(val_errors)]
    
def best_lasso(train_X,train_y,alpha_lst):
    '''
    This function performs cross-validation and then returns the 
    alpha which produces the lowest validation error.
    '''
    val_errors_1 = []
    for i in alpha_lst:
        lasso = Lasso(alpha = i)
        score_1 = -cross_val_score(lasso,train_X,train_y,scoring = 'neg_mean_squared_error',cv = 5)
        errors_1 = np.sum(score_1)
        val_errors_1.append(np.sqrt(errors_1))
    return alpha_lst[np.argmin(val_errors_1)]
    
def best_net(train_X,train_y,alpha_lst):
    '''
    This function performs cross-validation and then returns the 
    alpha which produces the lowest validation error.
    '''
    val_errors_2 = []
    for i in alpha_lst:
        net = ElasticNet(alpha = i)
        score_2 = cross_val_score(net,train_X,train_y,scoring = 'neg_mean_squared_error',cv = 5)
        errors_2 = np.sum(-score_2)
        val_errors_2.append(np.sqrt(errors_2))
    return alpha_lst[np.argmin(val_errors_2)]
    
    
def main():
    '''
    This function calls the functions and sets the training and testing set.
    It prints out a summary report of the RMSE of different models and 
    the best alpha of the model.
    '''
    train_X,train_y,test_X,test_y =  get_frames()
    alpha_ridge =  best_ridge(train_X,train_y,[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5])
    ridge = Ridge(alpha = alpha_ridge)
    alpha_lasso = best_lasso(train_X,train_y,[0.025,0.026,0.027,0.028,0.029,0.03,0.031,0.032,0.033,0.034,0.035])
    lasso =  Lasso(alpha = alpha_lasso)
    alpha_net = best_net(train_X,train_y,[0.035,0.036,0.037,0.038,0.039,0.04,0.041,0.042,0.043,0.044,0.045])
    net = ElasticNet(alpha = alpha_net)
    ols = LinearRegression()
    ols.fit(train_X,train_y)
    ridge.fit(train_X,train_y)
    lasso.fit(train_X,train_y)
    net.fit(train_X,train_y)
    ols_preds = ols.predict(test_X)
    ridge_preds = ridge.predict(test_X)
    lasso_preds = lasso.predict(test_X)
    net_preds = net.predict(test_X)
    print('Summary')
    print('\n')
    print('Ordinary Least Squares')
    print('RMSE: ' + str(np.sqrt(mean_squared_error(ols_preds,test_y))))
    print('\n')
    print('Ridge Regression')
    print('Best alpha value: ' + str(alpha_ridge))
    print('RMSE: ' + str(np.sqrt(mean_squared_error(ridge_preds,test_y))))
    print('\n')
    print('Lasso')
    print('Best alpha value: ' + str(alpha_lasso))
    print('RMSE: ' + str(np.sqrt(mean_squared_error(lasso_preds,test_y))))
    print('\n')
    print('Elastic Net')
    print('Best alpha value: ' + str(alpha_net))
    print('RMSE: ' + str(np.sqrt(mean_squared_error(net_preds,test_y))))
    print('\n')
    print('The best model is: Ridge Regression')
      
    
    
if __name__ == "__main__":
    main()