from hw2 import *
import hw2
import math
import unittest, sys, io, os, random, pickle
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
from compare_pandas import compare_lists, compare_los

"""
Rewrite these tests so that use only a portion of the corpus for
the sake of speed.
"""

"""
These tests depend on your code passing the previous tests, 
so get them working in the order that they are tested.

Making the classifier takes a long time, so in a lot of these
tests, I am doing odd things to bypass having to create a classifier
to access the methods.

Files needed:
hw2.py
get_data.py - necessary to download the data
labeled_data_test.pkl, labeled_data_train.pkl, word_counts_ld_train.pkl
word_probs_ld_train1.pkl, word_probs_ld_train1.pkl
main_out_correct.txt is NOT NEEDED
"""

class TestHw2(unittest.TestCase):
    
    def test_parse_line(self):
        line = 'List-Post: <mailto:exmh-users@example.com>'
        self.assertEqual('', LabeledData.parse_line(line))
        line = '    Message-ID:  <1029945287.4797.TMDA@deepeddy.vircio.com>'
        self.assertEqual('', LabeledData.parse_line(line))
        line = '  :    Message-ID:  <1029945287.4797.TMDA@deepeddy.vircio.com>'
        parsed = ':    Message-ID:  <1029945287.4797.TMDA@deepeddy.vircio.com>'
        self.assertEqual(parsed, LabeledData.parse_line(line))
        line = 'xxx  :    Message-ID:  <1029945287.4797.TMDA@deepeddy.vircio.com>'
        self.assertEqual('', LabeledData.parse_line(line))
        line = '18:19:04 Marking 1 hits' # oops! Not a header line.  Oh well.
        self.assertEqual('', LabeledData.parse_line(line))
        line = '  hey    Message-ID:  <1029945287.4797.TMDA@deepeddy.vircio.com>    '
        parsed = 'hey    Message-ID:  <1029945287.4797.TMDA@deepeddy.vircio.com>'
        self.assertEqual(parsed, LabeledData.parse_line(line))

    def test_parse_message(self):
        fp = io.StringIO("from: nowhere\nto: nobody\n\nWhat the...\nI'm confused")
        correct = "What the... I'm confused"
        with patch('builtins.open', side_effect=[fp]):
            self.assertEqual(correct, LabeledData.parse_message('dne.instance', 'dne.nada'))

        fp = io.StringIO("from nowhere\nto: nobody\n\nWhat the...\nI'm confused")
        correct = "What the... I'm confused"
        with patch('builtins.open', side_effect=[fp]):
            self.assertEqual(correct, LabeledData.parse_message('dne.instance', 'dne.nada'))

        fp = io.StringIO("from nowhere\nSubject:Hey Sucka\nto: nobody\n\nWhat the...\nI'm confused")
        correct = "Hey Sucka What the... I'm confused"
        with patch('builtins.open', side_effect=[fp]):
            self.assertEqual(correct, LabeledData.parse_message('dne.instance', 'dne.nada'))

        fp = io.StringIO("from nowhere\nSubject:Hey Sucka\nto: nobody\n\n" + 
            "    Subject: re: What the...\nI'm confused")
        correct = "Hey Sucka I'm confused"
        with patch('builtins.open', side_effect=[fp]):
            self.assertEqual(correct, LabeledData.parse_message('dne.instance', 'dne.nada'))
    
        fp = io.StringIO("from nowhere\nSubject: re: re: Hey Sucka\nto: nobody\n\n" + 
            "    Subject: re: What the...\nI'm confused")
        correct = "Hey Sucka I'm confused"
        with patch('builtins.open', side_effect=[fp]):
            self.assertEqual(correct, LabeledData.parse_message('dne.instance', 'dne.nada'))
       
        fp = io.StringIO("from nowhere\nSubject: Re: Re: Hey Sucka\nto: nobody\n\n" + 
            "    Subject: re: What the...\nI'm confused")
        correct = "Hey Sucka I'm confused"
        with patch('builtins.open', side_effect=[fp]):
            self.assertEqual(correct, LabeledData.parse_message('dne.instance', 'dne.nada'))
    
    def test_init_LabeledData(self):
        with patch('os.listdir', side_effect=[['hf1', 'hf2', 'hf3'], ['spf1', 'spf2']]):
            with patch('hw2.LabeledData.parse_message', side_effect=['ham1', 'ham2', 'ham3', 'spam1', 'spam2']):
                ld = LabeledData('hp', 'sp')
                self.assertEqual(['ham1', 'ham2', 'ham3', 'spam1', 'spam2'], ld.X)
                self.assertEqual([0, 0, 0, 1, 1], ld.y)
        with open('labeled_data_train.pkl', 'rb') as fp:
            correct = pickle.load(fp)
            student = LabeledData()
            self.assertTrue(compare_los(correct.X, student.X, skip_str_lens=True))
            self.assertTrue(compare_lists(correct.y, student.y))
        with open('labeled_data_test.pkl', 'rb') as fp:
            correct = pickle.load(fp)
            student = LabeledData('data/2003/easy_ham', 'data/2003/spam')
            self.assertTrue(compare_los(correct.X, student.X, skip_str_lens=True))
            self.assertTrue(compare_lists(correct.y, student.y))
        
    def test_tokenize(self):
        class HasStemmer:
            def __init__(self):
                self.stemmer = SnowballStemmer("english")
        nada = HasStemmer()

        correct = {'tell', 'ca', 'want', 's'}
        text = "I can't tell you 123, how567^%$ much I don't want to do this s#@!"
        self.assertEqual(correct, NaiveBayesClassifier.tokenize(nada, text))

        text = "WHAT IS GOING ON????"
        self.assertEqual(set(), NaiveBayesClassifier.tokenize(nada, text))

        correct = {'stopword', 'word'}
        text = "MUST FIND A WORD THAT ISN'T A STOPWORD.  WHAT IS IT?"
        self.assertEqual(correct, NaiveBayesClassifier.tokenize(nada, text))

        correct = {'swimmer', 'fast', 'fastest', 'swim', 'faster'} # this stemmer sucks
        text = "swim swimmer swimmer's swimming fast faster fastest"
        self.assertEqual(correct, NaiveBayesClassifier.tokenize(nada, text))

    def test_count_words(self):
        class HasLD:
            def __init__(self, ld):
                self.labeled_data = ld
                self.stemmer = SnowballStemmer("english")

        HasLD.tokenize = NaiveBayesClassifier.tokenize

        X = ['ha hsa, hb', 'sa, hsa$ sb', 'ha, hsa, hc']
        y = [0, 1, 0]
        ld = LabeledData(X=X, y=y)
        nada = HasLD(ld)

        correct = {'ha': [0, 2], 'hsa': [1, 2], 'hb': [0, 1], 'sa': [1, 0], 
            'sb': [1, 0], 'hc': [0, 1]}
        self.assertEqual(correct, NaiveBayesClassifier.count_words(nada))

        with open('word_counts_ld_train.pkl', 'rb') as fp:
            with open('labeled_data_train.pkl', 'rb') as fp2:
                correct = pickle.load(fp)
                ld = pickle.load(fp2)
                nada = HasLD(ld)
                self.assertEqual(correct, NaiveBayesClassifier.count_words(nada))

    def test_init_NaiveBayesClassifier(self):
        with open('labeled_data_train.pkl', 'rb') as fp:
            ld = pickle.load(fp)

        classifier = NaiveBayesClassifier(ld, max_words=25)
        with open('word_probs_ld_train1.pkl', 'rb') as fp:
            word_probs = pickle.load(fp)
        self.assertTrue(classifier.stemmer)
        self.assertEqual(ld, classifier.labeled_data)
        self.assertEqual(25, classifier.max_words)
        self.assertEqual(word_probs, classifier.word_probs)

        classifier = NaiveBayesClassifier(ld, 1)
        with open('word_probs_ld_train2.pkl', 'rb') as fp:
            word_probs = pickle.load(fp)
        self.assertEqual(50, classifier.max_words)
        self.assertEqual(word_probs, classifier.word_probs)

    def test_get_tokens(self):
        class HasMaxWords:
            def __init__(self, max_words=10):
                self.max_words = max_words

        random.seed(25)
        message = {'please', 'this', 'test', 'must', 'end'}
        correct = ['test', 'end', 'this']
        nada = HasMaxWords(3)
        self.assertEqual(correct, NaiveBayesClassifier.get_tokens(nada, message))

        correct = ['please', 'test', 'end', 'must', 'this']
        nada = HasMaxWords(7)
        self.assertEqual(correct, NaiveBayesClassifier.get_tokens(nada, message))

        correct = ['please', 'test']
        nada = HasMaxWords(2)
        self.assertEqual(correct, NaiveBayesClassifier.get_tokens(nada, message))

    def test_spam_probability(self):
        with open('labeled_data_train.pkl', 'rb') as fp:
            ld = pickle.load(fp)

        classifier = NaiveBayesClassifier(ld, max_words=25)
        message = ("This message is completely legit.  It is full of ham and good " +
                   "thoughts and ideas and excellent suggestions.")
        self.assertTrue(math.isclose(0.014279741503990657, classifier.spam_probability(message)))
    
    def test_classify(self):
        class HasSP:
            def __init__(self):
                self.gen = (i for i in [-1, 0, 0.49, 0.5, 0.51, 1, 2])

            def spam_probability(self, message):
                return next(self.gen)

        nada = HasSP()
        self.assertFalse(NaiveBayesClassifier.classify(nada, 'message'))
        self.assertFalse(NaiveBayesClassifier.classify(nada, 'message'))
        self.assertFalse(NaiveBayesClassifier.classify(nada, 'message'))
        self.assertTrue(NaiveBayesClassifier.classify(nada, 'message'))
        self.assertTrue(NaiveBayesClassifier.classify(nada, 'message'))
        self.assertTrue(NaiveBayesClassifier.classify(nada, 'message'))
        self.assertTrue(NaiveBayesClassifier.classify(nada, 'message'))

    def test_predict(self):
        class HasClassify:
            def __init__(self):
                self.gen = (b for b in [True, False, False, True])

            def classify(self, message):
                return next(self.gen)

        nada = HasClassify()
        X = ['m1', 'm2', 'm3', 'm4']
        correct = [True, False, False, True]
        self.assertEqual(correct, NaiveBayesClassifier.predict(nada, X))

    def test_main(self):
        correct = "[[2016  485]\n [   0  501]]\naccuracy: 83.84%\n"
        #correct = "[[2008  493]\n [   0  501]]\naccuracy: 83.58%\n"
        with io.StringIO() as buf, redirect_stdout(buf):
            random.seed(25)
            hw2.main()
            self.assertEqual(correct, buf.getvalue())


    """                
    def test_count_words(self):
        '''
        This is the most f***ed up test ever.  There must be an easier way.
        A much, much easier way.  Ok, found it, but keeping this because it's such a 
        ridiculous work around.
        '''
        class HasStemmer:
            def __init__(self):
                self.stemmer = SnowballStemmer("english")
        
        class HasLD:
            def __init__(self, ld):
                self.labeled_data = ld
            
            def tokenize(self, message):
                nada = HasStemmer()
                return hw2.NaiveBayesClassifier.tokenize(nada, message)

        X = ['ha hsa, hb', 'sa, hsa$ sb', 'ha, hsa, hc']
        y = [0, 1, 0]
        ld = LabeledData(X=X, y=y)
        nada = HasLD(ld)

        correct = {'ha': [0, 2], 'hsa': [1, 2], 'hb': [0, 1], 'sa': [1, 0], 
            'sb': [1, 0], 'hc': [0, 1]}
        self.assertEqual(correct, dict(hw2.NaiveBayesClassifier.count_words(nada)))
    """                

if __name__ == "__main__":
    test = unittest.defaultTestLoader.loadTestsFromTestCase(TestHw2)
    results = unittest.TextTestRunner().run(test)
    tests_run = results.testsRun
    failures = len(results.failures)
    errors = len(results.errors)
    sys.stdout = sys.__stdout__
    print('Correctness score = ', str((tests_run - errors - failures) / tests_run * 100) + '%')
