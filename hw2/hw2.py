"""
Name - Siddhant Bhardwaj
ISTA 331 HW1
Section Leader - Aleah M Crawford
Collaborators - Shivansh Singh Chauhan,Abhishek Agarwal,Vibhor Mehta, Sriharsha Madhira
"""

from sklearn.feature_extraction import stop_words
from nltk.stem import SnowballStemmer
from sklearn.metrics import confusion_matrix
import math,string,random,os

class LabeledData:
    def __init__(self,str1 = "data/2002/easy_ham",str2 = "data/2002/spam",X = None,y = None):
        '''
        '''
        lst_file = []
        lst_data = [] 
        if X is None:
            for file in os.listdir(str1):
                lst_data.append(self.parse_message(str1 + '/' + file))
                lst_file.append(0)
            for file in os.listdir(str2):
                lst_data.append(self.parse_message(str2 + '/' + file))
                lst_file.append(1)
            self.X = lst_data
            self.y = lst_file
        else:
            self.X = X
            self.y = y  
        
    def parse_message(self,fname):
        '''
        '''
        string  = ''
        with open(fname,errors = "ignore",encoding = "ascii") as file:
            result = []
            line = file.readline().strip()
            while line:
                if line.startswith ('Subject'):
                    remove = True
                    for token in line[8:].split():
                        if token.lower() == 're:' and remove:
                            continue
                        else:
                            result.append(token)
                            remove = False
                line = file.readline().strip()
            for line in file:
                res = LabeledData.parse_line(line)
                if res:
                    result.append(res)
        return ' '.join(result)
        
  
    @staticmethod    
    def parse_line(line):
        '''
        '''
        line = line.strip()
        pos = line.find(':')
        if pos <= 0:
           return line
        if len(line[:pos].split()) > 1:
            return line
        return ''
            
        
class NaiveBayesClassifier:
    def __init__(self,LabeledData,count = 0.5,max_words = 50):
        '''
        '''
        self.labeled_data =  LabeledData
        self.max_words =  max_words
        self.stemmer = SnowballStemmer('english')
        spam = 0
        ham = 0
        for num in self.labeled_data.y:
            if num == 0:
                ham += 1
            else:
                spam += 1
        self.word_probs = self.count_words()
        for i in self.word_probs:
           self.word_probs[i][0] = ( self.word_probs[i][0] + count ) / ( spam + (2 * count) )
           self.word_probs[i][1] = ( self.word_probs[i][1] + count ) / ( ham + (2 * count) )
   
   
    def tokenize(self,s):
        '''
        '''
        s = s.lower().replace("n't","")
        for char in string.punctuation + string.digits:
            s = s.replace(char,'')
        tokens = s.split()
        good_tokens =  set()
        for token in tokens:
            token = self.stemmer.stem(token)
            if token not in stop_words.ENGLISH_STOP_WORDS:
                good_tokens.add(token)
        return(good_tokens)
        
    def count_words(self):
        '''
        '''
        d = {}
        for i in range(len(self.labeled_data.X)):
            for token in (self.tokenize(self.labeled_data.X[i])):
                if token not in d:
                    d[token] = [0,0]
                if (self.labeled_data.y[i] == 0 ):
                    d[token][1] += 1
                else:
                    d[token][0] += 1
        return d
        
    def get_tokens(self,token_vector):
        '''
        '''
        token1 = sorted(token_vector)
        sample1 = random.sample(token1,min(self.max_words,len(token1)))
        return(sample1)
        
    def spam_probability(self,string):
        '''
        tokenized = self.tokenize(string)
        if tokenized.intersection(set(self.word_probs.keys())) == set():
            return 1
        else:
            spam = math.log(self.word_probs[i][0])
            ham = math.log(self.word_probs[i][1])
            random_tokens = self.get_tokens(list(tokenized))
            for token in random_tokens:
                if token in self.word_probs:
                    print(token)
        '''
        pass
        
    def classify(self,string):
        '''
        '''
        if self.spam_probability(string) >= 0.5:
            return True
        return False
    
        
    def predict(self,data_matrix):
        '''
        '''
        lst_predictions = []
        for message in data_matrix:
            lst_predictions.append(self.classify(message))
        return lst_predictions