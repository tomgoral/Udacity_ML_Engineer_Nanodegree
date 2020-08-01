
import argparse
import os
import numpy as np
import pandas as pd
import joblib

import subprocess as sb 
import sys
mypackage = 'sagemaker'
sb.call([sys.executable, "-m", "pip", "install", mypackage]) 


# estimators
from sklearn.neighbors           import KNeighborsRegressor
from sklearn.linear_model        import LinearRegression
from sklearn.ensemble            import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree                import DecisionTreeRegressor
from sklearn                     import svm, preprocessing
from sagemaker.sklearn.estimator import SKLearn



# model accuracy
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def model_fn(model_dir):
    # TODO instantiate a model from its artifact stored in model_dir
    model =  joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def predict_fn(input_data, model):
    # TODO apply model to the input_data, return result of interest
    return result










if __name__ =='__main__':

    print('\n1. EXTRACT ARGUMENTS:')
    parser = argparse.ArgumentParser()
    parser.add_argument('--choice', type=str, default ='rfr')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='test.csv')
    args, _ = parser.parse_known_args()

   
    print('2.         LOAD DATA:')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file),index_col =0)
    train_df = train_df.astype('float32')
    X_train = train_df[train_df.columns[1:]]
    y_train = train_df[train_df.columns[0]]    
    test_df = pd.read_csv(os.path.join(args.test, args.test_file),index_col =0)
    test_df = test_df.astype('float32')
    X_test = test_df[test_df.columns[1:]]
    y_test = test_df[test_df.columns[0]]
    

    print('3.        LOAD MODEL: ',args.choice)
    
    if args.choice == 'tree':        
        model = DecisionTreeRegressor(random_state=args.random_state)  # baseline
        
    elif args.choice == 'knn':        
        model =  KNeighborsRegressor()
       
    elif args.choice == 'rfr':        
        model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=args.min_samples_leaf,
                random_state = args.random_state,
                n_jobs=-1)
        
    elif args.choice == 'ada':        
        model =  AdaBoostRegressor(random_state=args.random_state)
    
    elif args.choice == 'linreg':
        model =  LinearRegression()  
    

    print('4.         FIT MODEL: ',args.choice)
    model.fit(X_train, y_train)
    
    
    print('5.        TEST MODEL: ',args.choice)
    preds_model     = model.predict(X_test)
    print('         mse: {:.4f}'.format(mean_squared_error(y_test,preds_model)))
    print('        rmse: {:.4f}'.format(np.sqrt(mean_squared_error(y_test,preds_model))))
    print('         mae: {:.4f}'.format(mean_absolute_error(y_test,preds_model)))
    print('          r2: {:.4f}'.format(r2_score(y_test,preds_model)))
     
        
    
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('6.    SAVED MODEL TO: ' + path)
  
