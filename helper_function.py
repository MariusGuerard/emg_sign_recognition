import time

import numpy as np


class ClosestMean():

    def __init__(self, geometry='euclid'):
        self.mean_dic = dict()
        self.result_dic = dict()
        if geometry == 'euclid':
            self.mean_func = lambda M_vec: np.mean(M_vec, axis=0)
            self.norm_func = lambda M1, M2: ((M1 - M2)**2).sum()
        self.geometry = geometry

    def fit(self, cov_train, y_train):
        self.labels = list(set(y_train))
        for label in self.labels:
            idx_label = np.where(y_train==label)[0]
            #cov_mean_label = np.mean(cov_train[idx_label], axis=0)
            cov_mean_label = self.mean_func(cov_train[idx_label])
            self.mean_dic[label] = cov_mean_label
            
    def predict(self, cov_test):
        yest_test = []
        for cov_sample in cov_test:
            dist_to_means = [self.norm_func(cov_sample, self.mean_dic[label]) for label in self.labels]
            yest_test.append(np.argmin(dist_to_means))
        return yest_test

    # To make it scikit-learn compatible.

    def get_params(self, deep=True):
        return {'geometry': self.geometry}
    
    def score(self, cov_test, y_test):
        yest_test = self.predict(cov_test)
        return accuracy(y_test, yest_test)
    
    
def vectorize_cov(cov_mat):
    """Transform a covariance matrix into an n(n+1)/2 elements vector.
    """
    lcov = len(cov_mat)
    covec = []
    for i in range(lcov):
        for j in range(i + 1):
            covec.append(cov_mat[i][j])
    return covec
    
def accuracy(yest, y):
    """Accuracy computed between estimated class and true class vectors.
    """
    yest = np.array(yest)
    y = np.array(y)
    return (yest == y).sum() / len(y)

def evaluate_on_kfold(model, X, y, kfold):
    """From a StratifiedKFold object, fit and test the model and return the accuracy results.
    Allow to compare different model on the same split and works with all models that have a 'fit' 
    and 'predict' method.
    """
    acc = []
    for train_indices, test_indices in kfold.split(X, y):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        model.fit(X_train, y_train)
        yest_test = model.predict(X_test)
        acc.append(accuracy(yest_test, y_test))
    return acc
    
def record_results_kfold(dic_data, dic_result, model, kfold, key_result, key_data='var'):
    dic_result[key_result] = {}
    dic_result[key_result]['acc'] = {}
    dic_result[key_result]['t'] = {}
    dic_result[key_result]['model'] = {}
    dic_result[key_result]['acc_avg'] = {}
    for key, data in dic_data.items():
        a = time.time()
        X = data[key_data]
        y = data['labels']
        acc = evaluate_on_kfold(model, X, y, kfold)
        time_eval = (time.time() - a) / kfold.n_splits
        dic_result[key_result]['acc'].setdefault(key, []).append(acc)
        dic_result[key_result]['t'].setdefault(key, []).append(time_eval)
        dic_result[key_result]['model'].setdefault(key, []).append(model)
        dic_result[key_result]['acc_avg'].setdefault(key, []).append(np.mean(acc))
        print(key, acc, np.mean(acc))
    dic_result[key_result]['acc_all'] = np.array([x for _, x in dic_result[key_result]['acc'].items()]).flatten()
    dic_result[key_result]['t_all'] = np.array([x for _, x in dic_result[key_result]['t'].items()]).flatten()
    return dic_result

def train_test_intersession(dic_data, dic_result, model, key_result, data_type='cov'):
    dic_result[key_result] = {'acc': {}, 't': {}, 'model': {}}
    score_list = []
    for train_key, data in dic_data.items():
        if 's1' in train_key:
            test_key = train_key[:-2] + '2*'
        elif 's2' in train_key:
            test_key = train_key[:-2] + '1*'
        data_train = dic_data[train_key]
        data_test = dic_data[test_key]
        X_train = data_train[data_type]
        X_test = data_test[data_type]
        y_train = data_train['labels']
        y_test = data_test['labels']          
        a = time.time()
        model.fit(X_train, y_train)
        yest_test = model.predict(X_test)
        score = accuracy(yest_test, y_test)
        time_eval = time.time() - a
        score_list.append(score)
        key_score = '{}-{}'.format(train_key, test_key)
        dic_result[key_result]['acc'].setdefault(key_score, []).append(score)
        dic_result[key_result]['t'].setdefault(key_score, []).append(time_eval)
        dic_result[key_result]['model'].setdefault(key_score, []).append(model)
        print(train_key, test_key, score)
    print('avg: {}'.format(np.mean(score_list)))
    return dic_result


def train_test_intersubjects(data_dic, dic_result, model, key_result, data_type='cov'):
    dic_result[key_result] = {}#{'acc': [], 't': {}, 'model': {}}
    keys = list(data_dic.keys())
    for train_key in keys:
        data_train = data_dic[train_key]
        X_train = data_train[data_type]
        y_train = data_train['labels']
        model.fit(X_train, y_train)            
        subject = train_key[:2]
        test_keys = [key for key in keys if subject not in key]
        for test_key in test_keys:
            data_test = data_dic[test_key]
            X_test = data_test[data_type]
            y_test = data_test['labels']      
            yest_test = model.predict(X_test)
            score = accuracy(yest_test, y_test)
            print(train_key, test_key, score)
            dic_result[key_result].setdefault('acc', []).append(score)
            dic_result[key_result].setdefault('model', []).append(model)
    print('avg: {}'.format(np.mean(dic_result[key_result]['acc'])))
    return dic_result