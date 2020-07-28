"""
Name - Siddhant Bhardwaj
ISTA 331 HW6
Section Leader -
Collaborators - Vibhor Mehta, Shivansh Singh Chauhan , Abhishek Agarwal,
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,mean_squared_error
from math import sqrt

def get_classification_frames():
    '''
    This is the get_classification_frames function.
    '''
    train_df = pd.read_csv('training.csv',usecols = ['class','b1','b2','b3','b4','b5','b6','b7','b8','b9'])
    test_df = pd.read_csv('testing.csv',usecols = ['class','b1','b2','b3','b4','b5','b6','b7','b8','b9'])
    return train_df,test_df
    
def get_X_and_y(frame):
    '''
    This is the get_X_and_y function.
    '''
    X = frame.drop(['class'],axis = 1)
    y = pd.Series(frame['class'])
    return X,y
    
def make_and_test_tree(train_X,train_y,test_X,test_y,depth):
    '''
    This is the make_and_test_tree function.
    '''
    dtree = DecisionTreeClassifier(max_depth = depth)
    dtree.fit(train_X,train_y)
    predictions = dtree.predict(test_X)
    cf = confusion_matrix(test_y,predictions)
    return cf
    
def get_regression_frame():
    '''
    This is the get_regression_frame function.
    '''
    df = pd.read_csv('bikes.csv',usecols = ['datetime','season','holiday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered','cnt'])
    print(len(df))
    return df
    
def get_regression_X_and_y(df):
    '''
    This is the get_regression_X_and_y function.
    '''
    training=np.random.choice(len(df),15000,replace=False)
    train_df=df.iloc[training]
    test_df=df.drop(training)
    predict=['season', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']; 
    train_X=train_df[predict]
    train_y=train_df['casual']
    test_X=test_df[predict]
    test_y=test_df['casual']
    return train_X,test_X,train_y,test_y   
    
def plot_confusion_matrix(train_X,train_y,test_X,test_y,dep):
    '''
    This is the plot_confusion_matrix function.
    '''
    cf = make_and_test_tree(train_X,train_y,test_X,test_y,depth=dep)
    plt.matshow(cf,cmap = plt.cm.gray)
    plt.title('Decision tree, max depth ' + str(dep))
    
    
def make_depth_plot(X,y,n,model = 'tree'):
    '''
    This is the make_depth_plot function.
    '''
    mean = []
    standard = []
    for i in range(1, n + 1):
        if model == 'tree':
            var1 = DecisionTreeRegressor(max_depth=i)
        if model == 'forest':
            var1 = RandomForestRegressor(n_estimators = 25,max_depth=i)
        score = cross_val_score(var1,X,y,scoring = 'neg_mean_squared_error',cv = 5)
        mean.append(score.mean())
        standard.append(score.std())
    plt.errorbar(range(1,n+1),mean,standard,lolims = False,xuplims = False,capsize = 2)
    plt.title('Neg MSE vs. depth, model_type = tree')
    var2 = np.array(mean)
    return var2.argmax() + 1 

def compare_regressors(train_X,train_y,test_X,test_y,list1):
    '''
    This is the compare_regressors function.
    '''
    predictions_dtree = list1[0].predict(train_X)
    predictions_dtree_1 = list1[0].predict(test_X)
    mean_sq_error_1 = mean_squared_error(train_y,predictions_dtree)
    predictions_rforest = list1[1].predict(train_X)
    predictions_rforest_1 = list1[1].predict(test_X)
    mean_sq_error_2 = mean_squared_error(train_y,predictions_rforest)
    R2_dtree = 1 - (mean_sq_error_1/np.var(train_y))
    R2_rforest = 1 - (mean_sq_error_2/np.var(train_y))
    print('-----------------------------------')
    print('Model type:   DecisionTreeRegressor')
    print('Depth:        ' + str(list1[0].tree_.max_depth))
    print('R^2:          ' + str(round(R2_dtree,4)))
    print('Testing RMSE: ' + str(round(sqrt(mean_squared_error(test_y,predictions_dtree_1)),4)))
    print('-----------------------------------')
    print('Model type:   RandomForestRegressor')
    print('Depth:        ' + str(list1[1].max_depth))
    print('R^2:          ' + str(round(R2_rforest,4)))
    print('Testing RMSE: ' + str(round(sqrt(mean_squared_error(test_y,predictions_rforest_1)),4)))  
            
            
def main():
    '''
    This is the main function.
    Also, press the 'X' sign to plot all the images.
    '''
    train_classify_df,test_classify_df = get_classification_frames()
    train_classify_X,train_classify_y = get_X_and_y(train_classify_df)
    test_classify_X,test_classify_y = get_X_and_y(test_classify_df)
    regression_df = get_regression_frame()
    train_X_regress,test_X_regress,train_y_regress,test_y_regress =  get_regression_X_and_y(regression_df)
    plot_confusion_matrix(train_classify_X,train_classify_y,test_classify_X,test_classify_y,1)
    plt.show()
    plot_confusion_matrix(train_classify_X,train_classify_y,test_classify_X,test_classify_y,5)
    plt.show()
    val1 = make_depth_plot(train_X_regress,train_y_regress,15,'tree')
    plt.show()
    val2 = make_depth_plot(train_X_regress,train_y_regress,15,'forest')
    plt.show()
    dtree = DecisionTreeRegressor(max_depth = val1)
    rforest = RandomForestRegressor(max_depth = val2)
    dtree.fit(train_X_regress,train_y_regress)
    rforest.fit(train_X_regress,train_y_regress)
    list2 = [dtree,rforest]
    compare_regressors(train_X_regress,train_y_regress,test_X_regress,test_y_regress,list2)
    
if __name__ == "__main__":
    main()