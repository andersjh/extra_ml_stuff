import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import tqdm

def evaluate(pipeline, X_train, X_test, y_train, y_test):
    '''
    Evaluate a pipeline on training and test datasets
    '''    
    pipeline.fit(X_train, y_train)
    y_train_hat = pipeline.predict(X_train)
    y_test_hat = pipeline.predict(X_test)
    train_f1 = f1_score(y_train_hat, y_train)
    train_acc = accuracy_score(y_train_hat, y_train)
    test_f1 = f1_score(y_test_hat, y_test)
    test_acc = accuracy_score(y_test_hat, y_test)

    print(f"========== Predictor: {type(pipeline).__name__} ==========")
    print(f"Training result: f1: {train_f1:.3f}, acc: {train_acc:.3f}")
    print(f"Test result: f1: {test_f1:.3f}, acc: {test_acc:.3f}")
    print()

# load dataset
df = pd.read_csv("transfusion.csv")
df.head(10)

# build X and y matrices
X = df.drop(['whether he/she donated blood in March 2007'], axis=1)
y = df[['whether he/she donated blood in March 2007']].values.reshape(-1)

# make sure there is no nan
# if there is nan, you need to deal with it, either by imputing or discarding
df.isnull().sum(axis = 0)

# split to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# try LogisticRegression to establish a baseline performance
pipeline = Pipeline([
    ('scale', StandardScaler()), # remember to scale first before feeding data into lgr
    ('lgr', LogisticRegression()),
])
evaluate(pipeline, X_train, X_test, y_train, y_test)

# try other predictors
evaluate(XGBClassifier(n_jobs=-1), X_train, X_test, y_train, y_test)
evaluate(LGBMClassifier(n_jobs=-1), X_train, X_test, y_train, y_test)
evaluate(RandomForestClassifier(n_jobs=-1), X_train, X_test, y_train, y_test)
evaluate(GradientBoostingClassifier(), X_train, X_test, y_train, y_test)

# RandomizedSearchCV on XGB
xgb_param_grid = {
    'n_estimators': [10, 20, 50, 100, 200, 300, 400],
    'max_depth': np.arange(5, 20),
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': np.arange(0.5, 1.0, 0.05),
    'min_child_weight': np.arange(1, 10),
    'colsample_bytree': np.arange(0.2, 1.0, 0.1),
    'gamma': [0, 0.001, 0.002, 0.003, 0.004, 0.005, 1e-2],
    'n_jobs': [-1]
}

predictor = XGBClassifier()
rs = RandomizedSearchCV(predictor, xgb_param_grid, cv=5, scoring='f1', n_jobs=-1, n_iter=100, verbose=1)
rs.fit(X_train, y_train)
evaluate(rs.best_estimator_, X_train, X_test, y_train, y_test)

# save model
with open(f'best_xgb_model.pickle', 'wb') as f:
    pickle.dump(rs.best_estimator_, f)

# evaluate model with kfold
kfold = KFold(n_splits=10)
results = cross_val_score(rs.best_estimator_, X, y, cv=kfold, n_jobs=-1)
print("Results: %.2f (%.2f) accuracy" % (results.mean(), results.std()))