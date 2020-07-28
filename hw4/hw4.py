"""
Name - Siddhant Bhardwaj
ISTA 331 HW4
Section Leader - Aleah Crawford
Collaborators - Vibhor Mehta, Shivansh Singh Chauhan , Abhishek Agarwal
"""

import numpy as np,pandas as pd,math
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def is_leap_year(yr):
    '''
    This code checks if a year is a leap year
    '''
    if str(yr)[-1] == '4' or str(yr)[-1] == '8' or str(yr)[-1] == '0':
        if yr[-2] % 2 == 0:
            return True
    else:
        if str(yr)[-1] == '2' or str(yr)[-1] == '6':
            if yr[-2] % 2 != 0:
                return True
    return False

def euclidean_distance(vector1,vector2):
    '''
    This function calculates the euclidean distance between vector1 and vector2
    '''
    sum = 0
    for i in range(len(vector1)):
        sum += (vector2[i] - vector1[i]) ** 2 
    return math.sqrt(sum)

def make_frame():
    '''
    This function will create a dataframe which will set the index to
    each date from 1987 to 2016
    '''
    frame = pd.read_csv('TIA_1987_2016.csv')
    frame['DateTime'] = pd.date_range(start = '01/01/1987',end = '31/12/2016')
    frame = frame.set_index(frame['DateTime'])
    frame.drop(columns  = ['DateTime'],axis = 1, inplace = True)
    return frame

def clean_dewpoint(frame):
    '''
    This function replaced the values for Dewpoint variable at March 10 and March 2011
    at 2010 with the average of all the other years on the same date.
    '''
    sum_2010_March_10 = 0
    sum_2010_March_11 = 0
    counter1 = 0
    counter2 = 0
    for i in frame.index:
        if str(i)[4:10] == '-03-10' and str(i)[0:4] != '2010':
            counter1 += 1
            if frame.loc[i,'Dewpt'] is not np.NaN:
                sum_2010_March_10 += frame.loc[i,'Dewpt']
            else:
                sum_2010_March_10 += 0
            
        if str(i)[4:10] == '-03-11' and str(i)[:4] != '2010':
            counter2 += 1
            if frame.loc[i,'Dewpt'] is not np.NaN:
                sum_2010_March_11 += frame.loc[i,'Dewpt']
            else:
                sum_2010_March_11 += 0
    avg_2010_March_10 = sum_2010_March_10 / counter1
    avg_2011_March_11 = sum_2010_March_11 / counter2
    frame.loc['2010-03-10','Dewpt'] =  avg_2010_March_10
    frame.loc['2010-03-11','Dewpt'] = avg_2011_March_11

def day_of_year(datetime_object):
    '''
    This function returns the day number in the year for the date concerned.
    For eb 29, it returns 366
    '''
    timestamp = datetime_object.strftime("%Y-%m-%d")
    if int(timestamp[:4]) % 4 == 0:
        if timestamp[4:] == '-02-29':
            return 366
        else:
            if int(timestamp[5:7]) > 2:
                return datetime_object.timetuple()[7] - 1
            else:
                return datetime_object.timetuple()[7]
    else:
        return datetime_object.timetuple()[7]

def climatology(frame):
    '''
    This function returns the mean of all columns for all days of the year
    for all years fri=om 1987 to 2016
    '''
    frame.reset_index(level =0,inplace = True)
    index_lst = []
    for i in range(len(frame)):
        index_lst.append(day_of_year(frame.loc[i,'index']))
    frame['index'] = index_lst
    df = frame.groupby(['index']).mean()
    df = df.iloc[:-1]
    return df

def scale(frame):
    '''
    This function scales the data in the dataframe using MinMaxScaler
    '''
    scaler = MinMaxScaler(copy = False)
    scaler.fit_transform(frame)

def get_initial_centroids(df,k):
    '''
    This function creates the initial centroids by setting all rows in the new_frame
    to i * (len(df) // k)
    '''
    new_frame = pd.DataFrame(columns=['Dewpt','AWS','Pcpn','MaxT','MinT'],index = [i for i in range(k)])
    for i in range(k):
        for j in range(len(df.columns)):
            new_frame.iloc[i,j] = df.iloc[i * (len(df) // k),j] 
    return new_frame

def classify(frame,feature_vec):
    '''
    This function classifies each feature vector to its closest centroid using euclidean distance.
    '''
    d = {}
    feature_vector = list(feature_vec)
    for i in frame.index:
        frame_vec = [frame.loc[i,j] for j in frame.columns]
        d[i] = euclidean_distance(frame_vec,feature_vec)
    return min(d, key=d.get)

def get_labels(frame,centroid_df):
    '''
    This function returns a Series containing the centroids closest to each feature vector 
    in the dataframe.
    '''
    d = {}
    for i in frame.index:
        d[i] = classify(centroid_df,frame.loc[i,:])
    return pd.Series(data = d)

def update_centroids(df,centroid_df,series_label):
    '''
    This function updates the centroids.
    '''
    centroid_df[:] = 0.0
    for day in df.index:
        centroid_df.loc[series_label[day]] += df.loc[day]
    counts = series_label.value_counts()
    for cent in centroid_df.index:
        centroid_df.loc[cent] /= counts[cent]

def k_means(df,k,centroids = None):
    '''
    This function returns the centroids and the final labels associated with
    after running the kmeans algorithm
    '''
    cents = get_initial_centroids(df,k)
    prev = cents.copy()
    labels = get_labels(df,cents)
    update_centroids(df,cents,labels)
    while not(cents.equals(prev)):
        prev = cents.copy()
        labels = get_labels(df,cents)
        update_centroids(df,cents,labels)
    return cents,labels

    