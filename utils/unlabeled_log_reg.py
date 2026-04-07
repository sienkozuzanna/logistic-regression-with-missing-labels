from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, balanced_accuracy_score
from utils.logistic_lasso_regression_fista import LogisticLassoRegressionFISTA
from utils.missing_schemas import MCAR, MNAR, MAR1, MAR2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


class UnlabeledLogReg:
    """
    A logistic regression model that can handle missing labels using either EM or kNN imputation.
    """

    def __init__(self, method="EM", max_em_iter=10, fista_params=None,
                 n_neighbors=5, label_type="proba"):
        
        """
        Parameters:
            - method: "EM" or "KNN"
            - max_em_iter: maximum number of EM iterations (only for EM method)
            - n_neighbors: number of neighbors for kNN imputation (only for KNN method)
            - label_type: "proba" or "hard" - whether to use predicted probabilities or hard labels for kNN imputation
        """
        
        if method not in ["EM", "KNN"]:
            raise ValueError("method must be 'EM' or 'KNN'")
        if label_type not in ["proba", "hard"]:
            raise ValueError("label_type must be 'proba' or 'hard'")
        
        self.method = method
        self.max_em_iter = max_em_iter
        self.n_neighbors = n_neighbors
        self.label_type = label_type

        self.fista = LogisticLassoRegressionFISTA(**(fista_params or {}))

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Fit the model on training data with missing labels.

        Parameters:
            - X_train: training features (numpy array)
            - y_train: training labels with missing values indicated by -1 (numpy array)
            - X_valid: validation features (numpy array)
            - y_valid: validation labels (numpy array)
            Returns:
            - self: fitted model
        """

        X = np.array(X_train)
        y = np.array(y_train)

        labeled_mask = (y != -1)
        unlabeled_mask = (y == -1)

        X_labeled, y_labeled = X[labeled_mask], y[labeled_mask]
        X_unlabeled = X[unlabeled_mask]

        # initial fit on labeled data to find best lambda
        self.fista.fit(X_labeled, y_labeled)
        self.fista.validate(X_valid, y_valid)
        self.best_lambda_ = self.fista.best_lambda_
        self.fista.lambdas = np.array([self.best_lambda_])

        if self.method == "KNN" and len(X_unlabeled) > 0:

            # kNN prediction for missing labels
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights="distance")
            knn.fit(X_labeled, y_labeled)

            if self.label_type == "hard":
                labels = knn.predict(X_unlabeled)
            elif self.label_type == "proba":
                labels = knn.predict_proba(X_unlabeled)[:, 1]
            else:
                raise ValueError("label_type must be 'proba' or 'hard'")

            # combine known and predicted labels
            X_full = np.vstack([X_labeled, X_unlabeled])
            y_full = np.concatenate([y_labeled, labels])

            # fit logistic regression
            self.fista.fit(X_full, y_full)
            self.fista.beta_ = self.fista.coef_paths_[self.best_lambda_]
            self.fista.intercept_ = self.fista.intercept_paths_[self.best_lambda_]

            return self
        
        if self.method == "EM" and len(X_unlabeled) > 0:

            for _ in range(self.max_em_iter):
                prev_beta = self.fista.beta_.copy()

                # E-step: predict probabilities for unlabeled data
                y_soft = self.fista.predict_proba(X_unlabeled)

                y_full = np.zeros(len(y))
                y_full[labeled_mask] = y_labeled
                y_full[unlabeled_mask] = y_soft

                # M-step: update model with new labels on new dataset
                self.fista.fit(X, y_full)

                self.fista.beta_ = self.fista.coef_paths_[self.best_lambda_]
                self.fista.intercept_ = self.fista.intercept_paths_[self.best_lambda_]

                if np.linalg.norm(self.fista.beta_ - prev_beta) < 1e-4:
                    break

        return self

    def predict(self, X_test):
        y_proba = self.fista.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_pred

    def evaluate(self, X_test, y_test, dataset_name, missing_schema):
        y_proba = self.fista.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)

        method_name = self.method
        if self.method == "KNN":
            method_name += f"_{self.label_type}"

        return {
            "method": method_name,
            "dataset": dataset_name,
            "missing_schema": missing_schema,
            "n_neighbors": self.n_neighbors if self.method == "KNN" else None,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
    
    
class OracleLogReg:
    """
    A logistic regression model trained only on the data with known labels (oracle method).
    """

    def __init__(self, fista_params=None):
        self.fista = LogisticLassoRegressionFISTA(**(fista_params or {}))

    def fit(self, X_train_full, y_train_full, X_valid, y_valid):
        if -1 in y_train_full:
            raise ValueError("OracleLogReg cannot be trained on data with missing labels")
        self.fista.fit(X_train_full, y_train_full)
        self.fista.validate(X_valid, y_valid)
        return self

    def predict_proba(self, X):
        return self.fista.predict_proba(X)
    
    def predict(self, X_test):
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_pred
    
    def evaluate(self, X_test, y_test, dataset_name, missing_schema):
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "method": "Oracle",
            "dataset": dataset_name,
            "missing_schema": missing_schema,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
    
    
class NaiveLogReg:
    """
    A logistic regression model trained only on the data with known labels, discarding all samples with missing labels (naive method).
    """
    
    def __init__(self, fista_params=None):
        self.fista = LogisticLassoRegressionFISTA(**(fista_params or {}))

    def fit(self, X_train_miss, y_train_miss, X_valid, y_valid):
        y = np.array(y_train_miss).flatten()
        labeled_mask = (y != -1)
        
        X_labeled = X_train_miss[labeled_mask]
        y_labeled = y[labeled_mask]
        
        self.fista.fit(X_labeled, y_labeled)
        self.fista.validate(X_valid, y_valid)
        return self

    def predict_proba(self, X):
        return self.fista.predict_proba(X)
    
    def predict(self, X_test):
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_pred
    
    def evaluate(self, X_test, y_test, dataset_name, missing_schema):
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "method": "Naive",
            "dataset": dataset_name,
            "missing_schema": missing_schema,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

'''
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
'''
