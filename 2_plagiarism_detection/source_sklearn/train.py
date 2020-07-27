from __future__ import print_function

import argparse
import os
import pandas as pd
import datetime
now = datetime.datetime.now()


## TODO: Import any additional libraries you need to define a model


from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier 

from sklearn.externals import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score
from sklearn.metrics import f1_score, make_scorer, mean_absolute_error, mean_squared_error
from sklearn.metrics import  precision_score, recall_score, r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    
    
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model
    ## TODO: Train the model
    
    random_state = 42
    
    '''
    
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                       n_estimators=10, random_state=42)
    accuracy:  0.9590686274509805                  
    
    
    BaggingClassifier(base_estimator=None, bootstrap=True, bootstrap_features=False,
                  max_features=3, max_samples=1.0, n_estimators=15, n_jobs=None,
                  oob_score=False, random_state=42, verbose=0,
                  warm_start=False)
    accuracy:  0.9653186274509805
    
      
    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                            max_depth=None, max_features=None, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, presort=False,
                            random_state=42, splitter='best')
    accuracy:  0.9340686274509805
    
    
    KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
    accuracy:  0.9590686274509805
    
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=2, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=30,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
    accuracy:  0.9653186274509805
    
       
    SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.6, kernel='rbf',
        max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001,
        verbose=False)
    accuracy:  0.9757352941176471
    
    
    
    
    '''
    
    svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True,
              probability=False, tol=0.001,cache_size=200, class_weight=None, verbose=False,
              max_iter=-1, decision_function_shape='ovr', random_state=random_state)
    
    parameters = {'C': [1,10,100,1000],'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    scoring      = make_scorer(fbeta_score, beta=0.5)
    grid_search  = GridSearchCV(estimator=svc, param_grid = parameters, scoring = scoring,cv=10)
    grid_search  = grid_search.fit(train_x, train_y)
    best_svc     = grid_search.best_estimator_
    model = best_svc

    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))