import unittest
from hw4 import *
from compare_pandas import compare_frames, compare_series

import pandas as pd
from datetime import datetime as dt

''' 
Files needed: 
compare_pandas.py, TIA_1987_2016.csv
make_frame.pkl, clean_frame.pkl, climatology.pkl, scale.pkl
gic1.pkl, gic5.pkl, gic6.pkl
centroids_1.pkl, labels_1.pkl, centroids_6.pkl, labels_6.pkl
'''

class TestAssignment3(unittest.TestCase):
    
    def test_make_frame(self):
        df_correct = pd.read_pickle('make_frame.pkl')
        self.assertTrue(compare_frames(df_correct, make_frame(), 0.001))
        
    def test_clean_dewpoint(self):
        df_correct = pd.read_pickle('clean_frame.pkl')
        df = pd.read_pickle('make_frame.pkl')
        self.assertIsNone(clean_dewpoint(df))
        self.assertTrue(compare_frames(df_correct, df, 0.001))
    
    def test_day_of_year(self):
        self.assertEqual(1, day_of_year(dt(2000, 1, 1)))
        self.assertEqual(29, day_of_year(dt(2000, 1, 29)))
        # new untested assertion
        self.assertEqual(59, day_of_year(dt(2000, 2, 28)))
        self.assertEqual(366, day_of_year(dt(2000, 2, 29)))
        self.assertEqual(365, day_of_year(dt(2000, 12, 31)))
        self.assertEqual(60, day_of_year(dt(2000, 3, 1)))
        self.assertEqual(60, day_of_year(dt(2001, 3, 1)))
       
    def test_climatology(self):
        df_correct = pd.read_pickle('climatology.pkl')
        df = pd.read_pickle('clean_frame.pkl')
        self.assertTrue(compare_frames(df_correct, climatology(df), 0.001))
        
    def test_scale(self):
        df_correct = pd.read_pickle('scale.pkl')
        df = pd.read_pickle('climatology.pkl')
        better_be_none = scale(df)
        self.assertTrue(compare_frames(df_correct, df, 0.001))
        self.assertIsNone(better_be_none)
        
    def test_get_initial_centroids(self):
        df_correct = pd.read_pickle('gic1.pkl')
        df = pd.read_pickle('scale.pkl')
        self.assertTrue(compare_frames(df_correct, get_initial_centroids(df, 1), 0.001))
        df_correct = pd.read_pickle('gic5.pkl')
        self.assertTrue(compare_frames(df_correct, get_initial_centroids(df, 5), 0.001))
        df_correct = pd.read_pickle('gic6.pkl')
        self.assertTrue(compare_frames(df_correct, get_initial_centroids(df, 6), 0.001))
        
    def test_classify(self):
        centroids = pd.read_pickle('gic6.pkl')
        index = ["Dewpt", "AWS", "Pcpn", "MaxT", "MinT"]
        fv = pd.Series([0, 0, 0, 0, 0], index)
        self.assertEqual(0, classify(centroids, fv))
        fv = pd.Series([100] * 5, index)
        self.assertEqual(4, classify(centroids, fv))
        fv = pd.Series([0, 0, 0, 0, 100], index)
        self.assertEqual(3, classify(centroids, fv))
        columns = ['one', 'two']
        data = [[3, 3], [3.5, 8], [8.5, 5.5]]
        centroids = pd.DataFrame(data, columns=columns)
        fv = pd.Series([4, 4], columns)
        self.assertEqual(0, classify(centroids, fv))
        fv = pd.Series([3, 9], columns)
        self.assertEqual(1, classify(centroids, fv))
        fv = pd.Series([10, 5], columns)
        self.assertEqual(2, classify(centroids, fv))
        
    def test_get_labels(self):
        columns = ['one', 'two']
        data_centroids = [[3, 3], [3.5, 8], [8.5, 5.5]]
        centroids = pd.DataFrame(data_centroids, columns=columns)
        data = [[2, 2], [4, 4], [4.5, 8.5], [2, 7], [9, 7], [7, 5]]
        sample = pd.DataFrame(data, columns=columns)
        correct = pd.Series([0, 0, 1, 1, 2, 2])
        self.assertTrue(compare_series(correct, get_labels(sample, centroids), 0.000001))
        
    def test_update_centroids(self):
        '''  This test depends on get_labels working correctly. '''
        columns = ['one', 'two']
        data_centroids_initial = [[3, 3], [3.5, 8], [8.5, 5.5]]
        centroids = pd.DataFrame(data_centroids_initial, columns=columns)
        data = [[2, 2], [4, 4], [4.5, 8.5], [2, 7], [9, 7], [7, 5]]
        sample = pd.DataFrame(data, columns=columns)
        labels = get_labels(sample, centroids)
        better_be_none = update_centroids(sample, centroids, labels)
        self.assertIsNone(better_be_none)
        data_centroids_final = [[3, 3], [3.25, 7.75], [8, 6]]
        centroids_correct = pd.DataFrame(data_centroids_final, columns=columns)
        self.assertTrue(compare_frames(centroids_correct, centroids, 0.000001))
        
        data_centroids_initial = [[0, 0], [8, 8]]
        centroids = pd.DataFrame(data_centroids_initial, columns=columns)
        data = [[1, 1], [3, 3], [5, 5], [5, 7], [7, 7], [7, 5]]
        sample = pd.DataFrame(data, columns=columns)
        labels = get_labels(sample, centroids)
        update_centroids(sample, centroids, labels)
        data_centroids_final = [[2, 2], [6, 6]]
        centroids_correct = pd.DataFrame(data_centroids_final, columns=columns)
        self.assertTrue(compare_frames(centroids_correct, centroids, 0.000001))
     
    def test_k_means(self):
        ct = pd.read_pickle('scale.pkl') # climatology
        centroids_correct = pd.read_pickle('centroids_1.pkl')
        labels_correct = pd.read_pickle('labels_1.pkl')
        centroids, labels = k_means(ct, 1)
        self.assertTrue(compare_frames(centroids_correct, centroids, 0.000001))
        self.assertTrue(compare_series(labels_correct, labels, 0.000001))
        centroids_correct = pd.read_pickle('centroids_6.pkl')
        labels_correct = pd.read_pickle('labels_6.pkl')
        centroids, labels = k_means(ct, 6)
        self.assertTrue(compare_frames(centroids_correct, centroids, 0.000001))
        self.assertTrue(compare_series(labels_correct, labels, 0.000001))
        
test = unittest.defaultTestLoader.loadTestsFromTestCase(TestAssignment3)
results = unittest.TextTestRunner().run(test)
print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 100) + ' / 100')
