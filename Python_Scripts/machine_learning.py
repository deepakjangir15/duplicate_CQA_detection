import xgboost as xgb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import helper as helper
import numpy as np
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


X_train = pd.read_pickle('dataset/X_train.pkl')
X_test = pd.read_pickle('dataset/X_test.pkl')
y_train = pd.read_pickle('dataset/y_train.pkl')
y_test = pd.read_pickle('dataset/y_test.pkl')

warnings.filterwarnings('ignore')

# Initialize the SGDClassifier
clf = SGDClassifier(loss='log', penalty='l2', random_state=101)

# Divide the dataset into small batches
batch_size = 1000
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    # Train the model on the current batch
    clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))

# Evaluate the model on the test set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Make predictions on the test data
y_pred_lr = clf.predict(X_test)

print('SGDClassifier/n')
helper.evaluate_all(y_test, y_pred_lr)
print('-------------------------------------------\n')

print('Random Forest Classifier/n')
warnings.filterwarnings('ignore')

# Create the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=101)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
helper.evaluate_all(y_test, y_pred)
print('-------------------------------------------\n')

print('XGBoost classifier/n')
# Create the XGBoost classifier
clf = xgb.XGBClassifier(n_estimators=100, max_depth=3,
                        learning_rate=0.1, random_state=101, verbose=True)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
helper.evaluate_all(y_test, y_pred)
print('-------------------------------------------\n')
