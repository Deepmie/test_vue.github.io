from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from sklearn.model_selection import train_test_split
import os.path
import numpy as np
import time
import pandas as pd
import pathos
from pyswarm import pso
import sys
import shap
import matplotlib.pyplot as plt
class MultilabelPredictor():
    """ Tabular Predictor for predicting multiple columns in table.
        Creates multiple TabularPredictor objects which you can also use individually.
        You can access the TabularPredictor for a particular label via: `multilabel_predictor.get_predictor(label_i)`

        Parameters
        ----------
        labels : List[str]
            The ith element of this list is the column (i.e. `label`) predicted by the ith TabularPredictor stored in this object.
        path : str, default = None
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
            Caution: when predicting many labels, this directory may grow large as it needs to store many TabularPredictors.
        problem_types : List[str], default = None
            The ith element is the `problem_type` for the ith TabularPredictor stored in this object.
        eval_metrics : List[str], default = None
            The ith element is the `eval_metric` for the ith TabularPredictor stored in this object.
        consider_labels_correlation : bool, default = True
            Whether the predictions of multiple labels should account for label correlations or predict each label independently of the others.
            If True, the ordering of `labels` may affect resulting accuracy as each label is predicted conditional on the previous labels appearing earlier in this list (i.e. in an auto-regressive fashion).
            Set to False if during inference you may want to individually use just the ith TabularPredictor without predicting all the other labels.
        kwargs :
            Arguments passed into the initialization of each TabularPredictor.

    """

    multi_predictor_file = 'multilabel_predictor.pkl'

    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, consider_labels_correlation=False,verbosity=2, **kwargs):
        if len(labels) < 2:
            raise ValueError("MultilabelPredictor is only intended for predicting MULTIPLE labels (columns), use TabularPredictor for predicting one label (column).")
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError("If provided, `problem_types` must have same length as `labels`")
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError("If provided, `eval_metrics` must have same length as `labels`")
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}  # key = label, value = TabularPredictor or str path to the TabularPredictor for this label
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i] : eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = self.path + "Predictor_" + label
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = eval_metrics[i]
            self.predictors[label] = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=path_i, verbosity=verbosity, **kwargs)

    def fit(self, train_data, tuning_data=None, **kwargs):
        """ Fits a separate TabularPredictor to predict each of the labels.

            Parameters
            ----------
            train_data, tuning_data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                See documentation for `TabularPredictor.fit()`.
            kwargs :
                Arguments passed into the `fit()` call for each TabularPredictor.
        """
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        if tuning_data is not None:
            tuning_data_og = tuning_data.copy()
        else:
            tuning_data_og = None
        save_metrics = len(self.eval_metrics) == 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            print(f"Fitting TabularPredictor for label: {label} ...")
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, **kwargs):
        """ Returns DataFrame with label columns containing predictions for each label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to make predictions for. If label columns are present in this data, they will be ignored. See documentation for `TabularPredictor.predict()`.
            kwargs :
                Arguments passed into the predict() call for each TabularPredictor.
        """
        return self._predict(data, as_proba=False, **kwargs)

    def predict_proba(self, data, **kwargs):
        """ Returns dict where each key is a label and the corresponding value is the `predict_proba()` output for just that label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to make predictions for. See documentation for `TabularPredictor.predict()` and `TabularPredictor.predict_proba()`.
            kwargs :
                Arguments passed into the `predict_proba()` call for each TabularPredictor (also passed into a `predict()` call).
        """
        return self._predict(data, as_proba=True, **kwargs)

    def evaluate(self, data, **kwargs):
        """ Returns dict where each key is a label and the corresponding value is the `evaluate()` output for just that label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to evalate predictions of all labels for, must contain all labels as columns. See documentation for `TabularPredictor.evaluate()`.
            kwargs :
                Arguments passed into the `evaluate()` call for each TabularPredictor (also passed into the `predict()` call).
        """
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            # print(f"Evaluating TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        """ Save MultilabelPredictor to disk. """
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(path=self.path+self.multi_predictor_file, object=self)
        print(f"MultilabelPredictor saved to disk. Load with: MultilabelPredictor.load('{self.path}')")

    @classmethod
    def load(cls, path):
        """ Load MultilabelPredictor from disk `path` previously specified when creating this MultilabelPredictor. """
        path = os.path.expanduser(path)
        if path[-1] != os.path.sep:
            path = path + os.path.sep
        return load_pkl.load(path=path+cls.multi_predictor_file)

    def get_predictor(self, label):
        """ Returns TabularPredictor which is used to predict this label. """
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            return TabularPredictor.load(path=predictor)
        return predictor

    def _get_data(self, data):
        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()

    def _predict(self, data, as_proba=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            # print(f"Predicting with TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(data, as_multiclass=True, **kwargs)
            data[label] = predictor.predict(data, **kwargs)
        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict
def run_secs(data,n,lb,ub):
    labels = ['Conversion Rate CH4','Hydrogen Yield']  # which columns to predict based on the others
    problem_types = ['regression','regression']  # type of each prediction problem (optional)
    eval_metrics = ['mean_absolute_error','mean_absolute_error']  # metrics used to evaluate predictions for each label (optional)
    save_path = 'agModels-predictEducationClass'  # specifies folder to store trained models (optional)
    df_train,df_test=train_test_split(data, test_size=0.2, random_state=0)
    test_dat=df_test.drop(['Conversion Rate CH4'],axis=1) 
    test_data=test_dat.drop(['Hydrogen Yield'],axis=1) 
    # test_data=test_data.drop(['Increment in methane yield'],axis=1) 
    X_train=df_train.iloc[:,0:-2]
    Y_train=df_train.iloc[:,-2:]
    X_valid=df_test.iloc[:,0:-2]
    Y_valid=df_test.iloc[:,-2:]
    #定义精度
    def print_accuracy(f):
        print("Root mean squared test error = {0}".format(np.sqrt(np.mean((f(X_valid) - Y_valid)**2))))
        time.sleep(0.5) # to let the print get out before any progress bars
    #定义标签名
    feature_names = df_train.columns[0:-2]
    # how many seconds to train the TabularPredictor for each label, set much larger in your applications!
    # multi_predictor60 = MultilabelPredictor(labels=labels, problem_types=problem_types, eval_metrics=eval_metrics, path=save_path)
    multi_predictor60=MultilabelPredictor.load('C:\\Users\\dell\\Desktop\\autogluonWEB\\agModels-predictEducationClass')
    # multi_predictor60=MultilabelPredictor.load('../agModels-predictEducationClass')
    for i in range (0,n):
        if i < 100:
            def obj(X):
                x2 = np.array(X[2])
                x3 = np.array(X[3])
                a = x2 / x3
                if 1 < a < 3.5:
                    X = pd.DataFrame(np.array([X[0], X[1], X[2], X[3], X[4], X[5]]).reshape(1, -1),
                                     columns=feature_names)
                    y_predict = multi_predictor60.predict(X)
                    CH4 = np.array(y_predict.iloc[:, 0])
                    H2 = np.array(y_predict.iloc[:, 1])
                    result = -(0.5 * CH4 + 0.5 * H2)
                else:
                    result = 0
                return result
            # lb = [850, 2, 10, 10, 500, 0.8]
            # ub = [950, 10, 80, 60, 5000, 1.5]
            xopt, fopt = pso(obj, lb, ub, ieqcons=[], maxiter=10)
            a = pd.DataFrame(xopt).T
            a.columns = feature_names
            y_opts = multi_predictor60.predict(a)
            pd.set_option('display.max_rows', None)#显示全部行
            pd.set_option('display.max_columns', None)#显示全部行
            b = pd.concat([a, y_opts], axis=1)
            return b
            # data_name =r"D:\machine learning\沼气制氢\CH4CO2\data\data"+str(i)+".csv"
            # b.to_csv(data_name)
        else:
            break

# 测试用例
# data=pd.DataFrame(np.random.normal(0,1,[100,8]))
# columns=['Temperature','Water Flow Rate','Volumetric Flow Rate CH4','Volumetric Flow Rate CO2','Space Velocity','Pressure','Conversion Rate CH4','Hydrogen Yield']
# data.columns=columns
# # lb_ub={'lb':[1,2,3,4,5,6],'ub':[2,3,4,5,6,7]}
# lb=[1,2,3,4,5,6]
# ub=[2,3,4,5,6,7]
# t=run_secs(data,2,lb,ub)
# print(t)
# def toTrain(n_iters,lb_ub,dataStr):
def toTrain(dataStr,n_iter,lb,ub):
    # train(0.2,60)
    data=dataStr
    # print(dataStr)
    init_data=data[2:-2]
    array=[i for i in range(len(init_data)) if init_data[i]==',' and init_data[i-1]==']' and init_data[i+1]=='[']
    num_array=[]
    number_array=[]
    for index in range(len(array)):
        if index==0:
            num_array.append(init_data[0:array[0]])
        else:
            num_array.append(init_data[array[index-1]+1:array[index]])
    num_array.append(init_data[array[index]+1:len(init_data)])
    # print(array)
    # opr Data
    lb=toArray(lb)
    ub=toArray(ub)
    n_iter=int(n_iter)
    # print(lb,type(lb))
    # print(ub,type(ub))
    # print(n_iter,type(n_iter))
    # print(number_array)
    for iarray in num_array:
        tar=iarray.replace(']','').replace('[','').split(',')
        new_tar=[float(i) for i in tar if len(i)>0]
        number_array.append(new_tar)
        # print(new_tar)
    pd_array=pd.DataFrame(number_array)
    colums=["Temperature","Water Flow Rate","Volumetric Flow Rate CH4","Volumetric Flow Rate CO2",
    "Space Velocity","Pressure","Conversion Rate CH4","Hydrogen Yield"]
    pd_array.columns=colums
    print(run_secs(pd_array,n_iter,lb,ub))
    # print(1)
def toArray(string):
    arra1=[int(i) for i in string.split(',')]
    return arra1

toTrain(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
# import os
# file_dir = "D:/machine learning/沼气制氢/CH4CO2/data/"
# files = os.listdir(file_dir)
# df1 = pd.read_csv(os.path.join(file_dir, files[0]))
# for e in files[1:]:
#     df2 = pd.read_csv(os.path.join(file_dir, e))
#     df1 = pd.concat((df1, df2), axis=0, join='inner')
# print(df1) 
# df1.to_csv("MOPSO_optimize_data.csv")