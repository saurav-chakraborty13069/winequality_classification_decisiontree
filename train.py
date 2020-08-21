import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
#from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from logger import App_Logger
import pickle

#log_writer = App_Logger()
# file_object = open("logs/TrainingLogs.txt", 'a+')


def get_data(log_writer, file_object):
    log_writer.log(file_object, 'Started getting the data')
    data = pd.read_csv('winequality_red.csv')
    return data

def check_data(data, log_writer, file_object):
    print(data.head())
    print(data.columns)
    print(data.info())
    print(data.describe())
    print(type(data))
    print(data.shape)
    print(data.isnull().sum())

def transform_data(data, log_writer, file_object):
    log_writer.log(file_object, 'Applying Scalar transformation')
    scalar = StandardScaler()
    data_scaled = scalar.fit_transform(data)
    return  data_scaled, scalar

def pca_data(data,log_writer, file_object):
    log_writer.log(file_object, 'Applying pca transformation')
    pca = PCA(n_components=10)
    new_data = pca.fit_transform(data)

    principal_x = pd.DataFrame(new_data,
                               columns=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9',
                                        'PC-10'])

    return principal_x, pca


def preprocess_data(data, log_writer, file_object):
    log_writer.log(file_object, 'Starting to Preprocess the data')
    q = data['fixed acidity'].quantile(0.99)
    data_cleaned = data[data['fixed acidity'] < q]
    q = data_cleaned['residual sugar'].quantile(0.95)
    data_cleaned = data_cleaned[data_cleaned['residual sugar'] < q]
    q = data_cleaned['chlorides'].quantile(0.95)
    data_cleaned = data_cleaned[data_cleaned['chlorides'] < q]
    q = data_cleaned['free sulfur dioxide'].quantile(0.95)
    data_cleaned = data_cleaned[data_cleaned['free sulfur dioxide'] < q]
    q = data_cleaned['total sulfur dioxide'].quantile(0.90)
    data_cleaned = data_cleaned[data_cleaned['total sulfur dioxide'] < q]
    q = data_cleaned['sulphates'].quantile(0.99)
    data_cleaned = data_cleaned[data_cleaned['sulphates'] < q]
    q = data_cleaned['alcohol'].quantile(0.98)
    data_cleaned = data_cleaned[data_cleaned['alcohol'] < q]

    check_data(data_cleaned, log_writer, file_object)
    return data_cleaned



def grid_search_data(clf, x_train, y_train, log_writer, file_object):
    log_writer.log(file_object, 'Starting the grid seach')
    log_writer.log(file_object, 'Setting up parameters')
    grid_param = {

        'max_depth': range(2, 32, 1),
        'min_samples_leaf': range(1, 10, 1),
        'min_samples_split': range(2, 10, 1),
        'splitter': ['best', 'random']
    }

    grid_search = GridSearchCV(estimator=clf,
                               param_grid=grid_param,
                               cv=5,
                               n_jobs=-1)
    log_writer.log(file_object, 'Fitting the Grid search Model')
    grid_search.fit(x_train, y_train)
    best_parameters = grid_search.best_params_
    log_writer.log(file_object, 'Best parameter for max depth is {}'.format(best_parameters['max_depth']))
    log_writer.log(file_object, 'Best parameter for min_samples_leaf is {}'.format(best_parameters['min_samples_leaf']))
    log_writer.log(file_object, 'Best parameter for min_samples_split is {}'.format(best_parameters['min_samples_split']))
    log_writer.log(file_object, 'Best parameter for splitter is {}'.format(best_parameters['splitter']))
    print("Best Grid Search Score is:   ", grid_search.best_score_)
    return best_parameters

def save_model(clf, scalar, pca, log_writer, file_object):

    # Writing different model files to file
    log_writer.log(file_object, 'Saving the models at location')
    with open('models/modelForPrediction.sav', 'wb') as f:
        pickle.dump(clf, f)

    with open('models/standardScalar.sav', 'wb') as f:
        pickle.dump(scalar, f)

    with open('models/modelpca.sav', 'wb') as f:
        pickle.dump(pca, f)



def train_data(log_writer):
    file_object = open("logs/TrainingLogs.txt", 'a+')
    log_writer.log(file_object, 'Start of Training')
    data = get_data(log_writer, file_object)
    log_writer.log(file_object, 'Received the data')
    check_data(data, log_writer, file_object)

    data_cleaned = preprocess_data(data, log_writer, file_object)
    log_writer.log(file_object, 'Preprocessing of data completed')
    X = data_cleaned.drop(['quality'], axis = 1)
    y = data_cleaned['quality']
    log_writer.log(file_object, 'Splitting the  data into X and y variables completed')
    x_transform, scaler = transform_data(X, log_writer, file_object)
    log_writer.log(file_object, 'Scalar transformation of features data completed' )
    principal_x, pca = pca_data(x_transform, log_writer, file_object)
    log_writer.log(file_object, 'Pca transformation of feature data completed ')
    x_train,x_test,y_train,y_test = train_test_split(principal_x, y, test_size = 0.30, random_state= 355)
    log_writer.log(file_object, 'Splitting of data into train and test completed')
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    log_writer.log(file_object, 'Training of Decision Tree classifier without best parameters')
    best_params = grid_search_data(clf, x_train, y_train, log_writer, file_object)
    log_writer.log(file_object, 'Grid search completed and received the best parameters ')
    clf = DecisionTreeClassifier(criterion = 'gini', max_depth =best_params['max_depth'],
                                 min_samples_leaf= best_params['min_samples_leaf'],
                                 min_samples_split= best_params['min_samples_split'],
                                 splitter = best_params['splitter'])
    clf.fit(x_train, y_train)
    log_writer.log(file_object, 'Fitting the Decision Tree Classifier with best parameters ')
    log_writer.log(file_object, 'Training complete')
    log_writer.log(file_object, '====================================')
    print('Current Classifier score is: ',clf.score(x_test, y_test))

    y_pred = clf.predict_proba(x_test)

    macro_roc_auc_ovo = roc_auc_score(y_test, y_pred, multi_class="ovo",
                                      average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_test, y_pred, multi_class="ovo",
                                         average="weighted")
    macro_roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class="ovr",
                                      average="macro")
    weighted_roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class="ovr",
                                         average="weighted")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
    save_model(clf, scaler, pca, log_writer, file_object)


