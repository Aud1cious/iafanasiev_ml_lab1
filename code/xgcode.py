import configparser
import os
filename = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config_path = filename + '/../config.ini'
print(config_path)
config.read(config_path)
print(config)


import pandas as pd
import os

db_train = pd.read_csv(filename + '/../data_bike/train.csv')
db_test = pd.read_csv(filename + '/../data_bike/test.csv')
db_test.head()
db_train.drop(['casual', 'registered'], axis=1, inplace=True)
db_train.columns
db_train['datetime_c'] =  pd.to_datetime(db_train['datetime'], format='%Y-%m-%d %H:%M:%S')
db_test['datetime_c'] =  pd.to_datetime(db_test['datetime'], format='%Y-%m-%d %H:%M:%S')
db_train.drop(['datetime'], axis=1, inplace=True)
db_test.drop(['datetime'], axis=1, inplace=True)

def datetime_2_features(df):
  
  df['year'] = df['datetime_c'].dt.year
  df['month'] = df['datetime_c'].dt.month
  df['day'] = df['datetime_c'].dt.day
  df['hour'] = df['datetime_c'].dt.hour
  df['min'] = df['datetime_c'].dt.minute
  df['sec'] = df['datetime_c'].dt.second
  df.drop(['datetime_c'], axis=1, inplace=True)

datetime_2_features(db_train)
datetime_2_features(db_test)
db_train.head()
X=db_train.iloc[:,db_train.columns!='count'].values
Y=db_train['count'].values
X_test = db_test[:].values

import xgboost as xg
xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)
#xgr.fit(X,Y)
xgr.load_model(filename + "/../model/model.txt")
print('Model Loaded')
xgr.save_model(filename + "/../" + config['MODEL']['path'])
print('Model Saved')

y_output=xgr.predict(X_test)
y_output

answer = pd.DataFrame({'count':(y_output)})
answer.to_csv(filename + '/../results/sub2.csv')

print('Code Completed')

