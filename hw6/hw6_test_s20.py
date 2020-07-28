import unittest, io, random, pickle as pkl
from contextlib import redirect_stdout
from hw6 import *
import pandas as pd, numpy as np, time, random
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from compare_pandas import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class TestAssignment6(unittest.TestCase):
    """
    def setUp(self):
        self.X = pd.read_pickle('X_correct.pkl')
        self.y = pd.read_pickle('y_correct.pkl')
        self.binned = pd.read_pickle('binned_ys_correct.pkl')
    """
    def test_get_classification_frames(self):
        train_correct = pd.read_pickle('train_frame_correct.pkl')
        test_correct = pd.read_pickle('test_frame_correct.pkl')
        train_out, test_out = get_classification_frames()
        self.assertTrue(compare_frames_str(train_correct, train_out))
        self.assertTrue(compare_frames_str(test_correct, test_out))
    
    def test_get_X_and_y(self):
        X = pd.read_pickle('X_correct.pkl')
        y = pd.read_pickle('y_correct.pkl')
        original = pd.read_pickle('train_frame_correct.pkl')
        #clean = original.copy()
        X_out, y_out = get_X_and_y(original)
        self.assertTrue(compare_frames_str(X, X_out))
        self.assertTrue(compare_series_str(y, y_out))
        #y_out += 1
        #self.assertTrue(compare_frames_str(original, clean))
    
    def test_make_and_test_tree(self):
        train_X = pd.read_pickle('train_X_correct.pkl')
        train_y = pd.read_pickle('train_y_correct.pkl')
        test_X = pd.read_pickle('test_X_correct.pkl')
        test_y = pd.read_pickle('test_y_correct.pkl')
        
        matrix_1_correct = pd.read_pickle('matrix_1_correct.pkl')
        matrix_3_correct = pd.read_pickle('matrix_3_correct.pkl')

        np.random.seed(23)
        matrix_1 = make_and_test_tree(train_X, train_y, test_X, test_y, 1)
        np.random.seed(23)
        matrix_3 = make_and_test_tree(train_X, train_y, test_X, test_y, 3)

        self.assertTrue((matrix_1_correct == matrix_1).all())
        self.assertTrue((matrix_3_correct == matrix_3).all())
        '''
        X_train = pd.get_dummies(X, prefix={'color':'color', 'finish':'finish', 'sun':'sun'})
        
        with io.StringIO() as buf, redirect_stdout(buf):
            np.random.seed(25)
            random.seed(25)
            tree = make_and_test(X_train, binned, "DecisionTreeClassifier", None, None, 'dtc_no_lim')
            self.assertTrue(type(tree) == DecisionTreeClassifier)
            correct = 'Tree accuracy DecisionTreeClassifier, all data max depth = inf: {}\n' + \
                      'Tree accuracy DecisionTreeClassifier, crss val max depth = inf: {}\n' + \
                      '----------------------------------------\n'
            retval = buf.getvalue()
        print(retval)
        all_data, crss_val = tuple(float(line[1].strip()) for line in [line.split(':') for line in retval.split('\n')][:-2]) 
        self.assertTrue(abs(0.99 - all_data) < 0.05)
        self.assertTrue(abs(0.73 - crss_val) < 0.05)
        correct = correct.format(all_data, crss_val)
        self.assertEqual(correct, retval)
        
        with io.StringIO() as buf, redirect_stdout(buf):
            np.random.seed(25)
            random.seed(25)
            tree = make_and_test(X_train, y, "RandomForestRegressor", None, None, 'rfr_no_lim', False)
            time.sleep(1)
            self.assertTrue(type(tree) == RandomForestRegressor)
            # Putting the vals later to make sure they match:
            correct = 'Tree accuracy RandomForestRegressor, all data max depth = inf: {}\n' + \
                      'Tree accuracy RandomForestRegressor, crss val max depth = inf: {}\n' + \
                      '----------------------------------------\n'
            retval = buf.getvalue()
        #print(retval)
        all_data, crss_val = tuple(float(line[1].strip()) for line in [line.split(':') for line in retval.split('\n')][:-2]) 
        self.assertTrue(abs(0.91 - all_data) < 0.05) 
        self.assertTrue(abs(0.5 - crss_val) < 0.05) 
        # In theory, I am now setting all of the seeds and I could just stick the numbers
        # straight into the output, but this works
        correct = correct.format(all_data, crss_val)
        self.assertEqual(correct, retval)
        '''

    def test_get_regression_frame(self):
        np.random.seed(42)
        correct = pd.read_pickle('correct_reg.pkl')
        reg_out = get_regression_frame()
        self.assertTrue(compare_frames_str(correct, reg_out))

    def test_get_regression_X_and_y(self):
        np.random.seed(42)
        correct_train_X = pd.read_pickle('correct_reg_train_X.pkl')
        correct_train_y = pd.read_pickle('correct_reg_train_y.pkl')
        correct_test_X = pd.read_pickle('correct_reg_test_X.pkl')
        correct_test_y = pd.read_pickle('correct_reg_test_y.pkl')
        data = get_regression_frame()
        train_X_out, test_X_out, train_y_out, test_y_out = get_regression_X_and_y(data)
        self.assertTrue(compare_frames_str(correct_train_X, train_X_out))
        self.assertTrue(compare_frames_str(correct_test_X, test_X_out))
        self.assertTrue(compare_series(correct_train_y, train_y_out))
        self.assertTrue(compare_series(correct_test_y, test_y_out))

    def test_make_depth_plot(self):
        np.random.seed(73)

        X = pd.read_pickle('small_X.pkl')
        y = pd.read_pickle('small_y.pkl')
        self.assertEqual(8,make_depth_plot(X, y, 15, 'tree'))
        self.assertEqual(12,make_depth_plot(X, y, 15, 'forest'))

    def test_compare_regressors(self):
        # self.maxDiff = None # not sure if this what finally made print(retval) below
        # work of if it was adding the print before it.
        train_X = pd.read_pickle('correct_reg_train_X.pkl')
        train_y = pd.read_pickle('correct_reg_train_y.pkl')
        test_X = pd.read_pickle('correct_reg_test_X.pkl')
        test_y = pd.read_pickle('correct_reg_test_y.pkl')
        
        np.random.seed(73)
        dtr = DecisionTreeRegressor(max_depth = 8)
        dtr.fit(train_X, train_y)
        rfr = RandomForestRegressor(n_estimators = 25, max_depth = 12)
        rfr.fit(train_X, train_y)
        with io.StringIO() as buf, redirect_stdout(buf):
            self.maxDiff = 1000

            self.assertIsNone(compare_regressors(train_X, train_y, test_X, test_y, [dtr, rfr]))
            correct = '-----------------------------------\n' + \
                      'Model type:   DecisionTreeRegressor\n' + \
                      'Depth:        8\n' + \
                      'R^2:          0.6629\n' + \
                      'Testing RMSE: 32.1867\n' + \
                      '-----------------------------------\n' + \
                      'Model type:   RandomForestRegressor\n' + \
                      'Depth:        12\n' + \
                      'R^2:          0.8338\n' + \
                      'Testing RMSE: 28.9576\n'
            retval = buf.getvalue()
            self.assertEqual(correct, retval)
            
    '''
    def test_main(self):
        with io.StringIO() as buf, redirect_stdout(buf):
            main()
            retval = buf.getvalue()
        for ch in '0123456789.':
            retval = retval.replace(ch, '')
        with open('main_out.pkl', 'rb') as fp:
            self.assertEqual(pkl.load(fp), retval)
    '''
    
    
test = unittest.defaultTestLoader.loadTestsFromTestCase(TestAssignment6)
results = unittest.TextTestRunner().run(test)
print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 100) + ' / 100')

























