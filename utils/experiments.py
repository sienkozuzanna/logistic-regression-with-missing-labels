import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from threadpoolctl import threadpool_limits

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from utils.missing_schemas import MCAR, MAR1, MAR2, MNAR
from utils.unlabeled_log_reg import UnlabeledLogReg, OracleLogReg, NaiveLogReg

def apply_missing_schema(X, y, schema_name, missing_rate):
    """
    Apply a missing data schema to the feature matrix.

    Parameters:
        X : pd.DataFrame - feature matrix.
        y : pd.Series - target labels.
        schema_name : str - missing data schema to apply, one of: 'MCAR', 'MAR1', 'MAR2', 'MNAR'.
        missing_rate : float - fraction of labels to mask as unlabeled, in range (0, 1).
    Returns:
        tuple - feature matrix (unchanged) and target labels with some values masked as unlabeled.
    Raises:
        ValueError - if schema_name is not one of the supported schemas.
    """

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
    """
    Run a single cross-validation fold for all methods.

    Splits the training fold into a labeled/unlabeled training set and a validation set,
    applies the missing data schema, scales features, and evaluates Oracle, Naive,
    KNN_hard, KNN_proba and EM models on the test set.

    Parameters:
        X : pd.DataFrame - full feature matrix.
        y : pd.Series - full target labels.
        train_idx : np.ndarray - indices of training samples for this fold.
        test_idx : np.ndarray - indices of test samples for this fold.
        schema_name : str - missing data schema to apply, one of: 'MCAR', 'MAR1', 'MAR2', 'MNAR'.
        missing_rate : float - fraction of training labels to mask as unlabeled.
        n_neighbors : int - number of neighbors used by KNN-based methods.
        dataset_name : str - dataset name passed to the evaluation method for logging purposes.
    Returns:
        list of dict - evaluation results, one per method, each containing metrics,
        dataset name, missing schema, missing rate and method name.
    """

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
    """
    Run the full experiment sequentially across all missing rates, schemas and folds.

    Iterates over all combinations of missing rates and missing data schemas
    using 2-fold stratified cross-validation.

    Parameters:
        X : pd.DataFrame - full feature matrix.
        y : pd.Series - full target labels.
        dataset_name : str - name of the dataset used for logging and result labeling. Default: 'SpamDataset'.
        missing_rates : list of float - missing rate values to evaluate. Default: [0.3, 0.5, 0.8].
    Returns:
        pd.DataFrame - evaluation results for all methods, folds, schemas and missing rates.
    """
    
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


def run_full_experiment_parallel(X, y, dataset_name="SpamDataset", missing_rates=[0.3, 0.5, 0.8]):
    """
    Run the full experiment in parallel across all missing rates, schemas and folds.

    Constructs a task list for all combinations of missing rates, schemas and 5-fold
    stratified cross-validation splits, then executes them in parallel using joblib
    with BLAS thread limits to avoid oversubscription.

    Parameters:
        X : pd.DataFrame - full feature matrix.
        y : pd.Series - full target labels.
        dataset_name : str - name of the dataset used for logging and result labeling. Default: 'SpamDataset'.
        missing_rates : list of float - missing rate values to evaluate. Default: [0.3, 0.5, 0.8].
    Returns:
        pd.DataFrame - evaluation results for all methods, folds, schemas and missing rates.
    """
    
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
    
    with threadpool_limits(limits=1, user_api='blas'):
        all_results_nested = Parallel(n_jobs=6, backend="loky", batch_size=1, verbose=10)(delayed(run_single_fold_wrapper)(X, y, t, dataset_name) for t in tasks)

    all_results = [item for sublist in all_results_nested for item in sublist]
    return pd.DataFrame(all_results)

def run_single_fold_wrapper(X, y, task, dataset_name):
    """
    Wrapper around run_single_fold for use with joblib Parallel.

    Unpacks a task dictionary and delegates to run_single_fold, then annotates each result with the fold index.

    Parameters:
        X : pd.DataFrame - full feature matrix.
        y : pd.Series - full target labels.
        task : dict - dictionary with keys: 'train_idx', 'test_idx', 'schema', 'c', 'fold_idx'.
        dataset_name : str - name of the dataset passed through to run_single_fold.
    Returns:
        list of dict - evaluation result dictionaries with fold index added.
    """

    res = run_single_fold(X, y, task['train_idx'], task['test_idx'], 
        task['schema'], task['c'], n_neighbors=5, dataset_name=dataset_name)
   
    for r in res:
        r['fold'] = task['fold_idx']
    return res