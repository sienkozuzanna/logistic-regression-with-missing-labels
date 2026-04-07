from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, balanced_accuracy_score
from utils.logistic_lasso_regression_fista import LogisticLassoRegressionFISTA
from utils.missing_schemas import MCAR, MNAR, MAR1, MAR2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


def generate_missing_label_data(X,y, missing_rate):
    #only for 1 split testing, not CV
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
    roc = roc_auc_score(y_test, y_proba)
    
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
    roc = roc_auc_score(y_test, y_proba)
    
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

def oracle_method(X_train, y_train, X_test, y_test, dataset_name):
    fista = LogisticLassoRegressionFISTA(
            max_iter=1000,
            stop_condition=1e-6)
    fista.fit(X_train=X_train, y_train=y_train)
    y_proba = fista.predict_proba(X_test = X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    row = pd.DataFrame([{
        "method": "oracle approach",
        "dataset": dataset_name,
        "missing_schema": '-',
        "n_neighbors": '-',
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "roc_auc": roc
    }])

    return row
    
    

def run_cv_averaged(X, y, dataset_name, missing_rate, n_neighbors=5, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    
    #ORACLE method
    metrics_oracle = []
    print("oracle method")#just to monitor progress
    k=1
    for train_idx, test_idx in skf.split(X, y):
        print(k)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        res = oracle_method(X_train, y_train, X_test, y_test, dataset_name)
        metrics_oracle.append(res)
        k+=1
        
    df_oracle = pd.DataFrame(metrics_oracle)
    results.append({
            "method": "oracle",
            "dataset": dataset_name,
            "missing_schema": '-',
            "n_neighbors": '-',
            "accuracy": df_oracle["accuracy"].mean(),
            "balanced_accuracy": df_oracle["balanced_accuracy"].mean(),
            "f1_score": df_oracle["f1_score"].mean(),
            "roc_auc": df_oracle["roc_auc"].mean()
        })
    
    #ACTUAL MISSING LABEL DATA
    for schema_name in ["MCAR", "MAR1", "MAR2", "MNAR"]:
        print(schema_name) #just to monitor progress
        
        #for metrics across folds
        metrics_unlabeled = []
        metrics_naive = []
        k=1
        for train_idx, test_idx in skf.split(X, y):
            print(k)
            X_train_df = X.iloc[train_idx]
            y_train_s = y.iloc[train_idx]
            X_test_df = X.iloc[test_idx]
            y_test_s = y.iloc[test_idx]

            if schema_name == "MCAR":
                X_miss_df, y_miss_s = MCAR(X_train_df, y_train_s, missing_rate)
            elif schema_name == "MAR1":
                X_miss_df, y_miss_s = MAR1(X_train_df, y_train_s, missing_rate)
            elif schema_name == "MAR2":
                X_miss_df, y_miss_s = MAR2(X_train_df, y_train_s, missing_rate)
            elif schema_name == "MNAR":
                X_miss_df, y_miss_s = MNAR(X_train_df, y_train_s, missing_rate)
                
            #convert to numpy
            X_miss = X_miss_df.values
            y_miss = y_miss_s.values
            X_test = X_test_df.values
            y_test = y_test_s.values

            scaler = MinMaxScaler()
            X_miss = scaler.fit_transform(X_miss)
            X_test = scaler.transform(X_test)    
            
            # run models
            res1 = unlabeled_log_reg_approach_1(
                X_miss, y_miss, X_test, y_test,
                n_neighbors=n_neighbors,
                dataset_name=dataset_name,
                missing_schema=schema_name
            )

            res2 = naive_approach(
                X_miss, y_miss, X_test, y_test,
                dataset_name=dataset_name,
                missing_schema=schema_name
            )

            metrics_unlabeled.append(res1)
            metrics_naive.append(res2)
            k+=1

        # convert to DataFrame
        df_unlabeled = pd.DataFrame(metrics_unlabeled)
        df_naive = pd.DataFrame(metrics_naive)

        # compute mean metrics
        results.append({
            "method": df_unlabeled["method"].iloc[0],
            "dataset": dataset_name,
            "missing_schema": schema_name,
            "n_neighbors": n_neighbors,
            "accuracy": df_unlabeled["accuracy"].mean(),
            "balanced_accuracy": df_unlabeled["balanced_accuracy"].mean(),
            "f1_score": df_unlabeled["f1_score"].mean(),
            "roc_auc": df_unlabeled["roc_auc"].mean()
        })

        results.append({
            "method": df_naive["method"].iloc[0],
            "dataset": dataset_name,
            "missing_schema": schema_name,
            "n_neighbors": None,
            "accuracy": df_naive["accuracy"].mean(),
            "balanced_accuracy": df_naive["balanced_accuracy"].mean(),
            "f1_score": df_naive["f1_score"].mean(),
            "roc_auc": df_naive["roc_auc"].mean()
        })

    return pd.DataFrame(results)