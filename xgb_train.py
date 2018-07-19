from collections import Counter
import pandas as pd
from sklearn import linear_model
import lightgbm as lgb
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

data_root = './'
train3 = pd.read_table(data_root+'SMPCUP2017_TrainingData_Task3.txt',sep='\001' ,names=['userID' ,'growthValue'])
stadata3 = pd.read_csv(data_root+'actStatisticData_new1.csv')
train3 = pd.merge(train3 ,stadata3 ,left_on='userID' ,right_on='userID' ,how='left')
x = np.array(train3.drop(['userID' ,'growthValue'] ,axis=1))
y = np.array(train3['growthValue'])
#submit
test3 = pd.read_table(data_root+'SMPCUP2017_TestSet_Task3.txt' ,sep='\001' ,names=['userID'])
test3 = pd.merge(test3 ,stadata3 ,left_on='userID' ,right_on='userID' ,how='left')

'''
param = {'max_depth':10,
             'eta': 0.22,
             'silent': 1,
             'objective': 'reg:tweedie',
             'booster': 'gbtree' ,
             'seed':10 ,
             'base_score':0.5 ,
             'eval_metric':'mae' ,
             'min_child_weight':1 ,
             'gamma':0.007 ,
             'tree_method':'hist' ,
             'tweedie_variance_power':1.54 ,
             'nthread':4,
                'tree_method': 'hist'
         }
num_round = 45
dtrain = xgb.DMatrix(x,label=y)
bst = xgb.train(param, dtrain, num_round)

x_t = xgb.DMatrix(np.array(test3.drop(['userID'] ,axis=1)))
y_t = bst.predict(x_t)
task3 = pd.DataFrame([test3['userID'] ,y_t]).T
task3 = task3.rename(columns={'userID':'userid' ,'Unnamed 0':'growthvalue'})
task3.to_csv('task3_final.txt' ,index=False ,sep=',')

'''
def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=80, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=100, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.04, min_child_weight=50, random_state=2020
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=10)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('submission.csv', index=False)
    return clf

test_x=np.array(test3.drop(['userID'] ,axis=1))
res=test3[['userID']]
model=LGB_predict(x,y,test_x,res)