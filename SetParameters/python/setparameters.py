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
import sys
import shap
import matplotlib.pyplot as plt
sys.path.append("C:\\Users\\dell\\AppData\\Roaming\\Python\\Python39\\site-packages\\autogluon")

# 定义多元预测机
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
            print(f"Evaluating TabularPredictor for label: {label} ...")
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
            print(f"Predicting with TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(data, as_multiclass=True, **kwargs)
            data[label] = predictor.predict(data, **kwargs)
        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict
# how many seconds to train the TabularPredictor for each label, set much larger in your applications!
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
# 定义可选参数的训练函数 
def train(test_size,time_limit,data):
    '''
    params:test_size,time_limit
    default:0.2,60
    '''
    # 定义标签，问题类型，精度判定标准以及保存路径
    labels = ['Conversion Rate CH4','Hydrogen Yield']  # which columns to predict based on the others
    problem_types = ['regression','regression']  # type of each prediction problem (optional)
    eval_metrics = ['mean_absolute_error','mean_absolute_error']  # metrics used to evaluate predictions for each label (optional)
    save_path = "C:\\Users\\dell\\Desktop\\autogluonWEB\\ImportData\\ag"  # specifies folder to store trained models (optional)
    #划分训练集和测试集
    df_train,df_test=train_test_split(data, test_size=test_size, random_state=0)
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
    #开始训练
    multi_predictor60 = MultilabelPredictor(labels=labels, problem_types=problem_types, eval_metrics=eval_metrics, path=save_path,consider_labels_correlation=False,verbosity=0)
    # data.to_excel('datas.xlsx')
    # data.to_csv('datas.csv')
    multi_predictor60.fit(df_train,time_limit=time_limit)
    multi_predictor60.save("C:\\Users\\dell\\Desktop\\autogluonWEB\\ImportData\\ag")
    # multi_predictor60.fit(df_train, time_limit=time_limit,verbosity = 0,presets='best_quality',auto_stack=True)
    print('Sucessful!')
    pd.set_option('display.max_rows', None)#显示全部行
    pd.set_option('display.max_columns', None)#显示全部行
    predictor_CH460 = multi_predictor60.get_predictor('Conversion Rate CH4')
    data1=predictor_CH460.leaderboard(silent=True)
    pd1=pd.DataFrame(data1)
    predictor_H260 = multi_predictor60.get_predictor('Hydrogen Yield')
    data2=predictor_H260.leaderboard(silent=True)
    pd2=pd.DataFrame(data2)
    print("**")
    print(pd1)
    print("**")
    print(pd2)
    X_train_summary = shap.kmeans(X_train,50)
    ag_wrapper = AutogluonWrapper(predictor_H260, feature_names)
    # print_accuracy(ag_wrapper.predict)
    explainer = shap.KernelExplainer(ag_wrapper.predict, X_train_summary)
    NSHAP_SAMPLES = 100  # how many samples to use to approximate each Shapely value, larger values will be slower
    N_VAL = 20  # how many datapoints from validation data should we interpret predictions for, larger values will be slower
    shap_values = explainer.shap_values(X_valid, nsamples=NSHAP_SAMPLES)
    shap.summary_plot(shap_values,X_valid,show=False)
    plt.savefig('C:\\Users\\dell\\Desktop\\autogluonWEB\\SetParameters\\img\\shap.png')
    
#SHAP的目的是解释每个特征对特定预测的贡献有多大。在这个回归的背景下，这相当于预测值与基线参考值的差异程度。我们首先在AutoGluon周围创建了一个封装类，允许它在shap包内被调用进行预测
class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict(X)
    # pd1.index=pd1.index+1
    # pd1_new=pd.concat([pd.DataFrame([pd1.columns]),pd1],axis=0)
    # pd1_new.columns=pd1_new.columns+1
    # pd1_new_new=pd.concat([pd.DataFrame(pd1_new.index),pd1_new],axis=1)
    # pd2.index=pd2.index+1
    # pd2_new=pd.concat([pd.DataFrame([pd2.columns]),pd1],axis=0)
    # pd2_new.columns=pd2_new.columns+1
    # pd2_new_new=pd.concat([pd.DataFrame(pd2_new.index),pd1_new],axis=1)
    # print('*')
    # print(pd1_new_new)
    # print(pd1)
# 测试用例
# data=pd.DataFrame(np.random.normal(0,1,[100,8]))
# columns=['Temperature','Water Flow Rate','Volumetric Flow Rate CH4','Volumetric Flow Rate CO2','Space Velocity','Pressure','Conversion Rate CH4','Hydrogen Yield']
# data.columns=columns
# print(data)
# train(0.2,60,data)
import sys
import time
import json
#预处理数据py
def toTrain(paramsStr,dataStr):
    obj=eval(paramsStr)
    print(obj['test_size'],obj['time_limit'])
    # train(0.2,60)
    data=dataStr
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
    print(pd_array)
    train(obj['test_size'],obj['time_limit'],pd_array)
    print(1)
#导入数据
toTrain(sys.argv[1],sys.argv[2])
# datas=pd.read_csv('../datas.csv')
# train(3,3,datas)