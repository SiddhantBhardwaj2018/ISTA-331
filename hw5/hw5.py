"""
Name - Siddhant Bhardwaj
ISTA 331 HW5
Section Leader - 
Collaborators - Vibhor Mehta, Shivansh Singh Chauhan , Abhishek Agarwal,
"""

import numpy as np,pandas as pd,string,math
from sklearn.feature_extraction import stop_words
from nltk.stem import SnowballStemmer

def dot_product(dict1,dict2):
	'''
    This function calculates dot product.
	'''
	sum = 0
	for key in dict1:
		if key in dict2:
			sum += dict2[key] * dict1[key]
	return sum

def magnitude(vector):
	'''
    This function calculates magnitude.
	'''
	sum1 = 0
	for key in vector:
		sum1 += vector[key] ** 2
	return math.sqrt(sum1)

def cosine_similarity(vector1,vector2):
	'''
    This function calculates cosine similarity.
	'''
	return dot_product(vector1,vector2) / (magnitude(vector1) * magnitude(vector2))

def get_text(filename):
	'''
    This function gets text from file.
	'''
	with open(filename,'r') as file:
		text =  file.read()
		text = text.replace("n't","").lower()
		for char in string.punctuation + string.digits:
			text = text.replace(char,"")
		return text

def vectorize(fname,stop_lst,stemmer1):
    '''
    This function vectorizes text.
    '''
    vector = {}
    text = get_text(fname)
    tokens = text.split()
    for token in tokens:
    	token1 = stemmer1.stem(token)
    	if token1 not in stop_lst and token1 != '':
    		if token1 not in vector:
    			vector[token1] = 1
    		else:
    			vector[token1] += 1
    return vector

def get_doc_freqs(lst_vectors):
    '''
    This function gets document frequencies.
    '''
    d = {}
    for i in range(len(lst_vectors)):
    	for key in lst_vectors[i]:
    		if key not in d:
    			d[key] = 1
    		else:
    			d[key] += 1
    return d	

def tfidf(lst_vectors):
    '''
    This is the tfidf function.
    '''
    lst1 = get_doc_freqs(lst_vectors)
    if len(lst_vectors) >= 100:
    	scale = 1
    else:
    	scale = 100 / len(lst_vectors)
    for key in lst_vectors:
    	for item in key:
    		key[item] = key[item] * (1 + math.log2(scale * (len(lst_vectors) / lst1[item])))

def get_similarity_matrix(lst_fnames,stopword_lst,stemmer1):
    '''
    This returns a dataframe which maps similarity based on tf-idf.
    '''
    df = pd.DataFrame(index = lst_fnames,columns = lst_fnames)
    lst = []
    for file in lst_fnames:
    	lst.append(vectorize(file,stopword_lst,stemmer1))
    tfidf(lst)
    for i in range(len(lst) - 1):
    	df.iloc[i,i] = 1
    	for j in range(i + 1,len(lst)):
    		cos1 =  cosine_similarity(lst[i],lst[j])
    		df.iloc[i,j] = df.iloc[j,i] = cos1
    	df.iloc[j,j] = 1
    return df

def main():
    '''
    This is the main function.
    '''
    stopword_lst =  {k for k in list(stop_words.ENGLISH_STOP_WORDS) + ['did', 'gone', 'ca']}
    stemmer1 = SnowballStemmer('english')
    lst = ['00000.txt','00001.txt','00002.txt','00003.txt','00004.txt','00005.txt']
    similarity = get_similarity_matrix(lst,stopword_lst,stemmer1)
    print(similarity)



