import unittest
import configparser
import os
import pandas as pd
import xgcode
import xgboost as xg

class TestPredict(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.dirname(os.path.abspath(__file__))
        self.config = configparser.ConfigParser()
        config_path = self.filename + '/../../config.ini'
        self.config.read(config_path)
        self.db_train = pd.read_csv(self.filename + '/../../data_bike/train.csv')
        self.db_test = pd.read_csv(self.filename + '/../../data_bike/test.csv')

    def test_smoke(self):
        self.assertEqual(self.db_train.shape[0] < self.db_test.shape[0], False)

    def test_data_pipeline(self):
        self.db_train['datetime_c'] =  pd.to_datetime(self.db_train['datetime'], format='%Y-%m-%d %H:%M:%S')
        xgcode.datetime_2_features(self.db_train)
        self.assertEqual(self.db_train.shape[1] - 4, 14)  

    def test_xg(self):
        #weights exitsts
        xgr=xg.XGBRegressor(max_depth=self.config['PARAMS']['max_depth'],
                      min_child_weight=self.config['PARAMS']['min_child_weight'], subsample=self.config['PARAMS']['subsample'], gamma=self.config['PARAMS']['gamma'],
                      colsample_bytree=self.config['PARAMS']['colsample_bytree'],) 
        xgr.load_model(self.filename + "/../../model/model.txt")
        


    



