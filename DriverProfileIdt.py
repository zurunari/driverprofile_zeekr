import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

dataset_idfr=2

# using dataset#3 here, since it has the tags already. No much work for encoding

path_file='datasets/test_motion_data.csv'

df = pd.read_csv(path_file) 

# ---------------------------------------
# Dataset overview ######
# freq = 1Hz (other datasets must be converted to this)\
# len = 3645
# dimensions: 3D acceleration, 3D gyro, sampled with MPI6050 module (enough precision)
# sample condition: sensor fixed on the trunk
# ---------------------------------------

X = df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']] # dataframe
y = df['Class']  # series

# FEATURE ENGINEERING

empirical_weights = [1,1,1.5,1,1,1] # intitial value, by Jeff
norm_coef = [1,1,1,1,1,1]
scaling_coef = [1,1,1,1,1,1]
X = np.linalg.multi_dot([X, empirical_weights, norm_coef, scaling_coef])

# train set 70% (default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='poly')  # optimize then
svm_model.fit(X_train, y_train)

# hyper param optm. ------------
param_grid = {
    'C': [0.1, 1, 10, 100],  # normalized parameter, default
    'gamma': [1, 0.1, 0.01, 0.001],  # kernel func coef, default
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # kernel type
}

cv_val = 5 # cross validation 
grid_search = GridSearchCV(svm_model, param_grid, refit=True, verbose=2, cv=cv_val)
cross_val_scores = cross_val_score(svm_model, X, y, cv=cv_val)

grid_search.fit(X_train, y_train)
print(f"Best Hyperparam for DPI: {grid_search.best_params_}")

y_pred = grid_search.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Cross-validation scores: {cross_val_scores}")
print(f"Average cross-validation score: {cross_val_scores.mean()}")
# -----------------------------

# Running on Azure student

job = command(
    code="./",  # directory where .py lies
    command="python train.py",  # command, here we don't have input params or env-dependent params
    inputs={"input_data": ClassifiedMotionData}, # data in Azure
    compute="workstation1",  # cluster of use
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1",  # env
)

ml_client.jobs.create_or_update(job)