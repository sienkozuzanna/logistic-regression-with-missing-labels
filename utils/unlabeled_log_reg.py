from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, balanced_accuracy_score
from utils.logistic_lasso_regression_fista import LogisticLassoRegressionFISTA
from utils.missing_schemas import MCAR, MNAR, MAR1, MAR2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

def generate_missing_label_data(X,y, missing_rate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,train_size=0.8)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #applying missing label schemas to train data
    X_mcar, y_mcar = MCAR(X_train, y_train, missing_rate)
    X_mar1, y_mar1 = MAR1(X_train, y_train, missing_rate)
    X_mar2, y_mar2 = MAR2(X_train, y_train, missing_rate)
    X_mnar, y_mnar = MNAR(X_train, y_train, missing_rate)
    missing_label_data = {   
        "MCAR": {
            "X_train": X_mcar,
            "y_train" : y_mcar
        },
        "MAR1": {
            "X_train": X_mar1,
            "y_train" : y_mar1
        },
        "MAR2": {
            "X_train": X_mar2,
            "y_train" : y_mar2
        },
        "MNAR": {
            "X_train": X_mnar,
            "y_train" : y_mnar
        },
        "test": {
            "X_test" : X_test,
            "y_test" : y_test
        }
    }
    return missing_label_data


def unlabeled_log_reg_approach_1(X_train, y_train, X_test, y_test, n_neighbors, dataset_name, missing_schema, label_type = 'proba'):
    #-1 => missing label
    X_missing = X_train[y_train == -1]
    
    y_not_missing = y_train[y_train != -1]
    X_not_missing = X_train[y_train != -1]
    
    #fitting a knn classifier to the not missing data to predict labels for the missing label instances
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights = 'distance')
    knn.fit(X_not_missing, y_not_missing)
    
    #predicting the labels for missing label instances
    if label_type == 'hard':
        labels = knn.predict(X_missing)
    elif label_type == 'proba':
        labels = knn.predict_proba(X_missing)[:, 1]
    else:
        raise ValueError("label_type needs to be equal to 'proba' or 'hard'")
    
    #connecting the two groups to train the logistic regression model
    X_combined = np.vstack([X_not_missing, X_missing])  # stack rows vertically
    y_combined = np.concatenate([y_not_missing, labels])
    
    #creating and fitting the logistic regression model
    fista = LogisticLassoRegressionFISTA(
        max_iter=1000,
        stop_condition=1e-6)
    fista.fit(X_train=X_combined, y_train=y_combined)
    
    #final prediction
    y_proba = fista.predict_proba(X_test = X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    
    row = pd.DataFrame([{
        "method": f"unlabeled_log_reg_knn_{label_type}",
        "dataset": dataset_name,
        "missing_schema": missing_schema,
        "n_neighbors": n_neighbors,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "roc_auc": roc
    }])

    return row

def naive_approach(X_train, y_train, X_test, y_test, dataset_name, missing_schema):
    #building the fista model on only the training data that does have a label
    y_not_missing = y_train[y_train != -1]
    X_not_missing = X_train[y_train != -1]
    fista = LogisticLassoRegressionFISTA(
        max_iter=1000,
        stop_condition=1e-6)
    
    fista.fit(X_train=X_not_missing, y_train=y_not_missing)
    
    y_proba = fista.predict_proba(X_test = X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    
    row = pd.DataFrame([{
        "method": f"naive approach",
        "dataset": dataset_name,
        "missing_schema": missing_schema,
        "n_neighbors": '-',
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "roc_auc": roc
    }])

    return row