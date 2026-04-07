import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from utils.missing_schemas import MCAR, MAR1, MAR2, MNAR
from utils.unlabeled_log_reg import UnlabeledLogReg, OracleLogReg, NaiveLogReg

def apply_missing_schema(X, y, schema_name, missing_rate):

    if schema_name == "MCAR":
        return MCAR(X, y, missing_rate)
    elif schema_name == "MAR1":
        return MAR1(X, y, missing_rate)
    elif schema_name == "MAR2":
        return MAR2(X, y, missing_rate)
    elif schema_name == "MNAR":
        return MNAR(X, y, missing_rate)
    else:
        raise ValueError(f"Unknown schema: {schema_name}")
    
def run_single_fold(X, y, train_idx, test_idx, schema_name, missing_rate, n_neighbors, dataset_name):

    X_train_full = X.iloc[train_idx]
    y_train_full = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

    X_miss_df, y_miss_s = apply_missing_schema(X_train_sub, y_train_sub, schema_name, missing_rate)

    scaler = MinMaxScaler()
    X_train_miss_scaled = scaler.fit_transform(X_miss_df.values)
    X_train_full_scaled = scaler.transform(X_train_sub.values)
    X_valid_scaled = scaler.transform(X_valid.values)
    X_test_scaled = scaler.transform(X_test.values)

    results = []
    methods_to_run = ["Oracle", "Naive", "KNN_hard", "KNN_proba", "EM"]

    for method in methods_to_run:
        if method == "Oracle":
            model = OracleLogReg()
            model.fit(X_train_full_scaled, y_train_sub.values, X_valid_scaled, y_valid.values)
        
        elif method == "Naive":
            model = NaiveLogReg()
            model.fit(X_train_miss_scaled, y_miss_s.values, X_valid_scaled, y_valid.values)
        
        elif method == "KNN_hard":
            model = UnlabeledLogReg(method="KNN", label_type="hard", n_neighbors=n_neighbors)
            model.fit(X_train_miss_scaled, y_miss_s.values, X_valid_scaled, y_valid.values)

        elif method == "KNN_proba":
            model = UnlabeledLogReg(method="KNN", label_type="proba", n_neighbors=n_neighbors)
            model.fit(X_train_miss_scaled, y_miss_s.values, X_valid_scaled, y_valid.values)
        
        elif method == "EM":
            model = UnlabeledLogReg(method="EM", n_neighbors=n_neighbors)
            model.fit(X_train_miss_scaled, y_miss_s.values, X_valid_scaled, y_valid.values)

        row = model.evaluate(X_test_scaled, y_test.values, dataset_name=dataset_name, missing_schema=schema_name)
        row['missing_rate'] = missing_rate
        results.append(row)

    return results

def run_full_experiment(X, y, dataset_name="SpamDataset", missing_rates=[0.3, 0.5, 0.8]):
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    all_results = []

    for c in missing_rates:
        print(f"\n[INFO] Testing missing rate (c) = {c}")
        for schema in ["MCAR", "MAR1", "MAR2", "MNAR"]:
            print(f"  -> Processing schema: {schema}")
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                fold_res = run_single_fold(X, y, train_idx, test_idx, schema, c, n_neighbors=5, dataset_name=dataset_name)
                
                for r in fold_res:
                    r['fold'] = fold_idx
                all_results.extend(fold_res)

    return pd.DataFrame(all_results)

#parallel version of the full experiment
def run_full_experiment_parallel(X, y, dataset_name="SpamDataset", missing_rates=[0.3, 0.5, 0.8]):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tasks = []
    for c in missing_rates:
        for schema in ["MCAR", "MAR1", "MAR2", "MNAR"]:
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                tasks.append({
                    'c': c,
                    'schema': schema,
                    'fold_idx': fold_idx,
                    'train_idx': train_idx,
                    'test_idx': test_idx
                })

    print(f"[INFO] Starting parallel execution for {len(tasks)} tasks...")

    all_results_nested = Parallel(n_jobs=-1, verbose=10)(delayed(run_single_fold_wrapper)(X, y, t, dataset_name) for t in tasks)
    all_results = [item for sublist in all_results_nested for item in sublist]

    return pd.DataFrame(all_results)

def run_single_fold_wrapper(X, y, task, dataset_name):
    res = run_single_fold(X, y, task['train_idx'], task['test_idx'], 
        task['schema'], task['c'], n_neighbors=5, dataset_name=dataset_name)
   
    for r in res:
        r['fold'] = task['fold_idx']
    return res