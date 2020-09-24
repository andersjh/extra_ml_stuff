import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import math
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# load dataset
df = pd.read_csv("transfusion.csv")
df.head(10)

# build X and y matrices
X = df.drop(['whether he/she donated blood in March 2007'], axis=1)
y = df[['whether he/she donated blood in March 2007']].values.reshape(-1)

# split to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# call TPOT and wait
tpot_clf = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1,
    max_time_mins=2, scoring='f1')
tpot_clf.fit(X_train, y_train)

# evaluate result
y_hat_test = tpot_clf.predict(X_test)
print(f'F1: {f1_score(y_test, y_hat_test)}')
print(f'Acc: {accuracy_score(y_test, y_hat_test)}')

# export model into python code
tpot_clf.export('tpot_model.py')