from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, balanced_accuracy_score
from utils.logistic_lasso_regression_fista import LogisticLassoRegressionFISTA
from utils.missing_schemas import MCAR, MNAR, MAR1, MAR2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


class UnlabeledLogReg:
    def __init__(self, approach='kNN', n_neighbors=15, max_iter=1000, stop_condition=1e-6, label_type='proba'):
        self.approach = approach
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.stop_condition = stop_condition
        self.label_type = label_type
        self.model = None

    def fit(self, X_train, y_train):
        if self.approach == 'kNN':
            X_missing = X_train[y_train == -1]
            y_not_missing = y_train[y_train != -1]
            X_not_missing = X_train[y_train != -1]

            # kNN prediction for missing labels
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance')
            knn.fit(X_not_missing, y_not_missing)

            if self.label_type == 'hard':
                labels = knn.predict(X_missing)
            elif self.label_type == 'proba':
                labels = knn.predict_proba(X_missing)[:, 1]
            else:
                raise ValueError("label_type must be 'proba' or 'hard'")

            # combine known and predicted labels
            X_combined = np.vstack([X_not_missing, X_missing])
            y_combined = np.concatenate([y_not_missing, labels])

            # fit logistic regression
            self.model = LogisticLassoRegressionFISTA(max_iter=self.max_iter, stop_condition=self.stop_condition)
            self.model.fit(X_train=X_combined, y_train=y_combined)
        else:
            raise NotImplementedError("Other approaches not implemented yet")

    def predict(self, X_test):
        y_proba = self.model.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_pred

    def evaluate(self, X_test, y_test, dataset_name, missing_schema):
        y_proba = self.model.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "method": f"unlabeled_log_reg_knn_{self.label_type}",
            "dataset": dataset_name,
            "missing_schema": missing_schema,
            "n_neighbors": self.n_neighbors,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }


class NaiveApproach:
    def __init__(self, max_iter=1000, stop_condition=1e-6):
        self.max_iter = max_iter
        self.stop_condition = stop_condition
        self.model = None

    def fit(self, X_train, y_train):
        X_not_missing = X_train[y_train != -1]
        y_not_missing = y_train[y_train != -1]
        self.model = LogisticLassoRegressionFISTA(max_iter=self.max_iter, stop_condition=self.stop_condition)
        self.model.fit(X_train=X_not_missing, y_train=y_not_missing)

    def predict(self, X_test):
        y_proba = self.model.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_pred

    def evaluate(self, X_test, y_test, dataset_name, missing_schema):
        y_proba = self.model.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "method": "naive approach",
            "dataset": dataset_name,
            "missing_schema": missing_schema,
            "n_neighbors": '-',
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
    


def run_cv_averaged(X, y, dataset_name, missing_rate, n_neighbors=5, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    # --- ORACLE METHOD ---
    print("oracle method")
    metrics_oracle = []
    k = 1
    for train_idx, test_idx in skf.split(X, y):
        print(k)
        X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

        # fit model on full labels
        oracle_model = NaiveApproach() #oracle is just naive but without any missing labels so we use the same class 
        oracle_model.fit(X_train, y_train)
        row = oracle_model.evaluate(X_test, y_test, dataset_name, missing_schema='-')
        metrics_oracle.append(row)
        k += 1

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

    # --- MISSING LABEL DATA ---
    for schema_name in ["MCAR", "MAR1", "MAR2", "MNAR"]:
        print(schema_name)
        metrics_unlabeled = []
        metrics_naive = []
        k = 1

        for train_idx, test_idx in skf.split(X, y):
            print(k)
            X_train_df, y_train_s = X.iloc[train_idx], y.iloc[train_idx]
            X_test_df, y_test_s = X.iloc[test_idx], y.iloc[test_idx]

            # generate missing labels
            if schema_name == "MCAR":
                X_miss_df, y_miss_s = MCAR(X_train_df, y_train_s, missing_rate)
            elif schema_name == "MAR1":
                X_miss_df, y_miss_s = MAR1(X_train_df, y_train_s, missing_rate)
            elif schema_name == "MAR2":
                X_miss_df, y_miss_s = MAR2(X_train_df, y_train_s, missing_rate)
            elif schema_name == "MNAR":
                X_miss_df, y_miss_s = MNAR(X_train_df, y_train_s, missing_rate)

            # convert to numpy and scale
            X_miss = MinMaxScaler().fit_transform(X_miss_df.values)
            X_test_scaled = MinMaxScaler().fit_transform(X_test_df.values)
            y_miss = y_miss_s.values
            y_test = y_test_s.values

            # --- Unlabeled LogReg ---
            unlabeled_model = UnlabeledLogReg(approach='kNN', n_neighbors=n_neighbors)
            unlabeled_model.fit(X_miss, y_miss)
            row_unlabeled = unlabeled_model.evaluate(X_test_scaled, y_test, dataset_name, schema_name)
            metrics_unlabeled.append(row_unlabeled)

            # --- Naive Approach ---
            naive_model = NaiveApproach()
            naive_model.fit(X_miss, y_miss)
            row_naive = naive_model.evaluate(X_test_scaled, y_test, dataset_name, schema_name)
            metrics_naive.append(row_naive)

            k += 1

        # compute mean metrics per method
        df_unlabeled = pd.DataFrame(metrics_unlabeled)
        df_naive = pd.DataFrame(metrics_naive)

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
