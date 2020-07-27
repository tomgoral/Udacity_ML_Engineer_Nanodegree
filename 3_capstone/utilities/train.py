from __future__ import print_function
import argparse
import os
import pandas as pd
import datetime
now = datetime.datetime.now()


#  preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


# estimators
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble     import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree         import DecisionTreeRegressor
from sklearn              import svm, preprocessing


# model accuracy
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



from sklearn.externals import joblib



def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model



if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()

    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    train_y = train_data.iloc[:,0]     # response
    train_x = train_data.iloc[:,1:]    # features

    
    random_state = 42
    tree           = DecisionTreeRegressor(random_state=random_state)  # benchmark
