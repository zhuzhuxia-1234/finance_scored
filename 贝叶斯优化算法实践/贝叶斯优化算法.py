# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:16:00 2019

使用 Hyperopt 进行参数调优
@author: zhuxibing
"""

##一个简单的例子，在-2到 2之间随机取数，求（x-^）2的最小值
from hyperopt import fmin, tpe, hp
best = fmin(fn=lambda x: (x-1)**2,space=hp.uniform('x', -2, 2),algo=tpe.suggest,max_evals=100)
print(best)

'''
hp.choice(label, options) 其中options应是 python 列表或元组。

hp.normal(label, mu, sigma) 其中mu和sigma分别是均值和标准差。

hp.uniform(label, low, high) 其中low和high是范围的下限和上限。
'''
from hyperopt.pyll.stochastic import sample

space={
       'x':hp.uniform('x',0,1),
       'y':hp.normal('y',0,1),
       'name':hp.choice('name',['alice','bob']),
       }
#从space中去一个样本
print(sample(space))

'''
通过Trial捕获信息，如果能看到hyperopt黑匣子内发生了什么是极好的。Trials对象使我们能够做到这一点
'''
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

fspace = {'x': hp.uniform('x', -5, 5)}

def f(params):
    x = params['x']
    val = x**2
    return {'loss': val, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)

print('best:', best)
print('trials:')
for trial in trials.trials:
    print(trial)
    
    
########################
#    实战
########################

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4knn = {'n_neighbors': hp.choice('n_neighbors', range(1,100))}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:{}'.format(best))



from sklearn.tree import DecisionTreeClassifier

def hyperopt_train_test_1(params):
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4dt = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
}

def f(params):
    acc = hyperopt_train_test_1(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)

print('best:{}'.format(best))


import numpy as np
import lightgbm as lgb
from hyperopt import tpe,Trials,hp,fmin,STATUS_OK

tpe_algorithm = tpe.suggest
bayes_trials = Trials()

train_set = lgb.Dataset(X,y)
N_FOLDS=5
MAX_EVALS=100
def objective(params, n_folds=N_FOLDS):
   cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=10000,
                       early_stopping_rounds=100, metrics='auc', seed=50)

   best_score = max(cv_results['auc-mean'])
   loss = 1 - best_score
   return {'loss': loss, 'params': params, 'status': STATUS_OK}


space = {
   'class_weight': hp.choice('class_weight', [None, 'balanced']),
   'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
   'subsample': hp.uniform('gdbt_subsample', 0.5, 1), 
   'num_leaves': hp.choice('num_leaves', range(30, 150, 1)),
   'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
   'subsample_for_bin': hp.choice('subsample_for_bin', range(20000, 300000, 20000)),
   'min_child_samples': hp.choice('min_child_samples', range(20, 500, 5)),
   'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
   'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
   'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

best = fmin(fn = objective, space = space, algo = tpe.suggest, 
           max_evals = MAX_EVALS, trials = bayes_trials)

print(best)






