import time

import numpy as np


class MDM():
    """Classifier based on Minimum Distance to Mean (MDM)
    Note: For now just the MDM on euclidean geometry is implemanted 
    (MDM on riemann is already implemented on other libraries as pyriemann)
    """
    def __init__(self, geometry='euclid'):
        """Initiate an instance of the class on the specified geometry.
        the choice of the geometry impacts the computation of the mean and of the norm.
        """
        self.mean_dic = dict()
        self.result_dic = dict()
        if geometry == 'euclid':
            self.mean_func = lambda M_vec: np.mean(M_vec, axis=0)
            self.norm_func = lambda M1, M2: ((M1 - M2)**2).sum()
        self.geometry = geometry

    def fit(self, X_train, y_train):
        """Compute the mean of all the samples corresponding to each class.
        
        Args:
            X_train (np.array): Training samples with dimension N_samples x N_features.
            y_train (np.array): Training labels with dimension N_samples.
        """
        self.labels = list(set(y_train))
        for label in self.labels:
            idx_label = np.where(y_train==label)[0]
            cov_mean_label = self.mean_func(X_train[idx_label])
            self.mean_dic[label] = cov_mean_label
            
    def predict(self, X_test):
        """Return the labels that corresponds to the closest mean to each sample.
        
        Args:
            X_test (np.array): Testing sample with dimension N_samples x N_features
            
        Returns:
            yest_test (np.array): Estimation of the test label with dimension N_samples.
        """
        yest_test = []
        for X_sample in X_test:
            dist_to_means = [self.norm_func(X_sample, self.mean_dic[label]) for label in self.labels]
            yest_test.append(np.argmin(dist_to_means))
        return yest_test

    # Methods to make the class 'scikit-learn compatible'.
    def get_params(self, deep=True):
        return {'geometry': self.geometry}
    
    def score(self, X_test, y_test):
        yest_test = self.predict(X_test)
        return accuracy(y_test, yest_test)
    
    
def vectorize_cov(cov_mat):
    """Transform a covariance matrix into an N(N+1)/2 elements vector. 
    As covariance matrix are symmetric, all the information is contained in this vector.
    
    Args:
        cov_mat (np.array): Square matrix of dimension N x N
        
    Returns:
        covec (np.array): Vector containing the elements on the bottom left of the diagonal(included). 
        Dimension N(N+1)/2
    """
    lcov = len(cov_mat)
    covec = []
    for i in range(lcov):
        for j in range(i + 1):
            covec.append(cov_mat[i][j])
    return covec
    
    
def accuracy(yest, y):
    """Accuracy computed between estimated class and true class vectors.
    
    Args:
        yest (np.array): estimated labels vector
        y (np.array): true labels vector.
    """
    yest = np.array(yest)
    y = np.array(y)
    return (yest == y).sum() / len(y)


def evaluate_on_kfold(model, X, y, kfold):
    """From a StratifiedKFold object, fit and test the model and return the accuracy results.
    Allow to compare different models on the same split and works with all models that have a 'fit' 
    and 'predict' method.
    
    Args:
        model (Obj): any instance of a class with predict and append methods.
        X (np.array): Input matrix of dim N_samples x N_features.
        y (np.array): Output labels of dimension N_samples.
        kfold (Obj): any instance of a class with a split method (taking X, and y as argument).
        
    Returns:
        acc ([float]): accuracy measured on every split of the kfold.
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
    """Pipeline for the intra-session experiments. For each sessions, fit the model on one fold and
    validate on other folds and store the results in 'dic_result'.
    
    Args:
        dic_data (dict): contains the data in different shape (raw, covariance, variance,...)
        dic_result (dict): contains the accuracy, time of execution and trained model for each experiment.
        model (Obj): model to test.
        kfold (Obj): kfold to test the model on.
        key_result (str): key to use for storing in dic_result (refers to the model name)
        key_data (str): key to select the shape of the input data (covariance, variance,...)
        
    Returns:
        dic_data (dict): contains the data in different shape (raw, covariance, variance,...)
    """
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
    """Pipeline for the inter-session experiments. For each session couple, fit the model on one session and
    validate it on another session before storing the results in 'dic_result'.
        
    Args:
        dic_data (dict): contains the data in different shape (raw, covariance, variance,...)
        dic_result (dict): contains the accuracy, time of execution and trained model for each experiment.
        model (Obj): model to test.
        key_result (str): key to use for storing in dic_result (refers to the model name)
        data_type (str): key to select the shape of the input data (covariance, variance,...)
        
    Returns:
        dic_data (dict): contains the data in different shape (raw, covariance, variance,...)
    """
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
    """Pipeline for the inter-subject experiments. For each session couple that are drawn from different users, 
    fit the model on one session and validate it on another session. Then store the results in 'dic_result'.
    
     Args:
        dic_data (dict): contains the data in different shape (raw, covariance, variance,...)
        dic_result (dict): contains the accuracy, time of execution and trained model for each experiment.
        model (Obj): model to test.
        key_result (str): key to use for storing in dic_result (refers to the model name)
        data_type (str): key to select the shape of the input data (covariance, variance,...)
        
    Returns:
        dic_data (dict): contains the data in different shape (raw, covariance, variance,...)
    """
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