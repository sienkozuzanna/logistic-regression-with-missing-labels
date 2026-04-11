import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

class LogisticLassoRegressionFISTA:

    SUPPORTED_MEASURES = {"recall", "precision", "f1", "balanced_accuracy", "roc_auc", "avg_precision"}

    def __init__(self, lambdas=None, measure="roc_auc", max_iter=1000, stop_condition=1e-6, step_size=None):
        """
        Initializes the LogisticLassoRegressionFISTA model.

        Parameters:
            lambdas: list or np.ndarray - array of lambda values to try during fitting (if None, a default logarithmic grid will be used)
            measure: str - evaluation metric to use for selecting the best lambda during validation (default is "roc_auc")
            max_iter: int - maximum number of iterations for the FISTA optimization (default is 1000)
            stop_condition: float - threshold for stopping the optimization based on parameter change (default is 1e-6)
            step_size: float or None - step size for gradient updates (if None, it will be computed based on Lipschitz constant estimation)
        """
       
        if lambdas is None:
            self.lambdas = np.logspace(-4, 1, 20)
        else:
            self.lambdas = np.atleast_1d(lambdas)

        self.measure = measure
        self.max_iter = max_iter
        self.stop_condition = stop_condition
        self.step_size = step_size

        self.coef_paths_ = {} 
        self.intercept_paths_ = {}

        self.beta_ = None
        self.intercept_ = None
        self.best_lambda_ = None

    def _sigmoid(self, x):
        """
        Calculates standard sigmoid:
            1 / (1 + exp(-x))

        Parameters:
            x: np.ndarray - input value

        Returns:
            np.ndarray - result of applying sigmoid function
        """

        return 1.0 / (1.0 + np.exp(-x))
    
    def _compute_default_step_size(self, X):
        """
        Computes default step size based on an estimate of the Lipschitz constant of the gradient.

        Parameters:
            X: pd.DataFrame - explanatory variables

        Returns:
            float - computed step size
        """
        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])   # including inptercept in computing step size - without it it's too large
        spectral_norm = np.linalg.norm(X_aug, 2)
        L = (spectral_norm ** 2) / (4.0 * n)
        return 1.0 / (L + 1e-12)

    def _gradient_step_L1_part(self, x, threshold):
        """
        Used to handle the non-differentiable L1 penalty.
        How it works:
            - large positive values are reduced toward zero,
            - large negative values are increased toward zero,
            - small values with size <= threshlod are set to zero

        Parameters:
            x: np.ndarray - input value
            threshold: float - threshold value 

        Returns:
            np.ndarray - result after applying threshold for each element of the array
        """

        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def _compute_gradient(self, X, y, beta, intercept):
        """
        Calculates gradient for the smooth part of the loss-likelihood function, e.i. the part without L1.

        Parameters:
            X: pd.DataFrame - explanatory variables
            y: pd.Series - target variable
            beta: pd.Series - model parameters
            intercept: float - intercept (beta0)

        Returns:
            tuple - gradient with respect to beta & gradient with respect to intercept
        """
        n = X.shape[0]

        z = X @ beta + intercept
        p = self._sigmoid(z)

        error = p - y

        grad_beta = (X.T @ error) / n
        grad_intercept = np.sum(error) / n

        return grad_beta, grad_intercept

    def _compute_logistic_loss(self, X, y, beta, intercept):
        """
        Calculates mean logistic loss without the L1 penalty.

        Parameters:
            X: pd.DataFrame - explanatory variables
            y: pd.Series - target variable
            beta: pd.Series - model parameters
            intercept: float - intercept (beta0)

        Returns:
            float - mean logistic loss value
        """
        z = X @ beta + intercept
        p = self._sigmoid(z)

        eps = 1e-15
        p = np.clip(p, eps, 1-eps)

        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss
    
    def _objective(self, X, y, beta, intercept):
        """
        Calculates full objective function:
            logistic_loss + lambda * |beta|

        Parameters:
            X: pd.DataFrame - explanatory variables
            y: pd.Series - target variable

        Returns:
            float - value of the full objective function
        """
        loss = self._compute_logistic_loss(X=X, y=y, beta=beta, intercept=intercept)
        L1_penalty = self.lambda_ * np.sum(np.abs(beta))
        return loss + L1_penalty
    
    def _compute_metric(self, y_true, proba, measure):
        """
        Computes the specified evaluation metric based on true labels and predicted probabilities.

        Parameters:
            y_true: np.ndarray - true binary labels
            proba: np.ndarray - predicted probabilities for class 1
            measure: str - name of the metric to compute

        Returns:
            float - computed metric value
        """
        y_pred = (proba >= 0.5).astype(int)
        if measure == "recall":
            return recall_score(y_true, y_pred, zero_division=0)
        elif measure == "precision":
            return precision_score(y_true, y_pred, zero_division=0)
        elif measure == "f1":
            return f1_score(y_true, y_pred, zero_division=0)
        elif measure == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        elif measure == "roc_auc":
            return roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
        elif measure == "avg_precision":
            return average_precision_score(y_true, proba)
        raise ValueError(f"Unknown measure: {measure}. Choose from {self.SUPPORTED_MEASURES}")

    def _fit_single(self, X, y):
        """
        Fits the logistic regression model with L1 regularization for a single value of lambda using FISTA optimization.

        Parameters:
            X: pd.DataFrame - explanatory variables
            y: pd.Series - target variable
        Returns:
            tuple - fitted parameters (beta, intercept)
        
        """

        _, p = X.shape

        if self.step_size is None:
            step = self._compute_default_step_size(X=X)
        else:
            step = self.step_size

        beta = np.zeros(p)
        intercept = 0.0

        # clue of FISTA: beta_momentum, intercept_momentum - accelerated point used to compute gradient
        ### w FISTA nie patrzymy tylko na ostatnii punkt, tylko tez na kierunek, w ktorym szlismy wczesniej ??
        beta_momentum = beta.copy()         
        intercept_momentum = intercept      
        t = 1.0 

        self.objective_history_ = []

        for i in range(self.max_iter):
            beta_prev = beta.copy()
            intercept_prev = intercept

            grad_beta, grad_intercept = self._compute_gradient(
                X=X, 
                y=y, 
                beta=beta_momentum, 
                intercept=intercept_momentum,
            )

            beta_temp = beta_momentum - step * grad_beta
            intercept_temp = intercept_momentum - step * grad_intercept

            beta = self._gradient_step_L1_part(
                x=beta_temp, 
                threshold=step * self.lambda_,
            )
           
            intercept = intercept_temp

            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))

            beta_momentum = beta + ((t - 1.0) / t_new) * (beta - beta_prev)
            intercept_momentum = intercept + ((t - 1.0) / t_new) * (intercept - intercept_prev)

            t = t_new

            self.objective_history_.append(self._objective(X, y, beta, intercept))

            param_change = np.linalg.norm(beta - beta_prev) + abs(intercept - intercept_prev)
            if param_change < self.stop_condition:
                self.n_iter_ = i+1
                self.beta_ = beta
                self.intercept_ = intercept
                break
            
        else:
            self.n_iter_ = self.max_iter
            self.beta_ = beta
            self.intercept_ = intercept
            
        return beta, intercept


    def fit(self, X_train, y_train):
        """
        Fits the logistic regression model with L1 regularization using FISTA optimization.

        Parameters:
            X_train: pd.DataFrame - training explanatory variables
            y_train: pd.Series - training target variable
        """
        X = np.array(X_train)
        y = np.array(y_train)

        for lam in self.lambdas:
            self.lambda_ = lam
            beta, intercept = self._fit_single(X, y)
            
            self.coef_paths_[lam] = beta
            self.intercept_paths_[lam] = intercept
        
        return self
    
    
    def predict_proba(self, X_test):
        """
        Returns predicted probabilities for class 1.
        """
        if self.beta_ is None:
            raise RuntimeError("Model parameters are not set. You must call validate() before predict_proba() to select the best lambda.")
        X = np.array(X_test)
        scores = X @ self.beta_ + self.intercept_
        return self._sigmoid(scores)
    
    
    def validate(self, X_valid, y_valid, measure=None):
        """
        Evaluates the model on the validation set using the specified measure.

        Parameters:
            X_valid: pd.DataFrame - validation explanatory variables
            y_valid: pd.Series - validation target variable
            measure: str - name of the metric to compute (optional, if not provided, uses the measure specified during initialization)

        Returns:
            float - computed metric value on the validation set
        """

        measure = measure if measure is not None else self.measure
        if measure not in self.SUPPORTED_MEASURES:
            raise ValueError(f"Unsupported measure: {measure}. Choose from {self.SUPPORTED_MEASURES}")
        
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)

        self.val_scores_ = {}

        for lam in self.lambdas:
            beta = self.coef_paths_[lam]
            intercept = self.intercept_paths_[lam]
            
            proba = self._sigmoid(X_valid @ beta + intercept)
            score = self._compute_metric(y_valid, proba, measure)
            self.val_scores_[lam] = score

        self.best_lambda_ = max(self.val_scores_, key=self.val_scores_.get)

        self.lambda_ = self.best_lambda_
        self.beta_ = self.coef_paths_[self.best_lambda_]
        self.intercept_ = self.intercept_paths_[self.best_lambda_]

        return self.val_scores_[self.best_lambda_]

# ------------------------- PLOTS -------------------------

    def plot(self, X_valid, y_valid, measure = None):
        """
        Plot the evaluation measure as a function of lambda.
        
        Parameters:
            X_valid: validation features
            y_valid: validation labels
            measure: metric to plot (optional, default is self.measure)
        """

        measure = measure if measure is not None else self.measure
        measures_names = {
            "roc_auc": "ROC AUC",
            "f1": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
            "balanced_accuracy": "Balanced Accuracy",
            "avg_precision": "Average Precision"
        }
        display_name = measures_names.get(measure, measure.replace('_', ' ').title())

        scores = []

        for lam in self.lambdas:
            beta = self.coef_paths_[lam]
            intercept = self.intercept_paths_[lam]
            proba_predictions = self._sigmoid(X_valid @ beta + intercept)
            score = self._compute_metric(y_valid, proba_predictions, measure)
            scores.append(score)

        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        plt.semilogx(self.lambdas, scores, marker='o', markersize=5, linestyle='-', 
                 linewidth=2, color='#2c7bb6', label=f'Validation {display_name}')
        
        if self.best_lambda_ is not None:
            best_score = self.val_scores_[self.best_lambda_]
            plt.axvline(x=self.best_lambda_, color='#d7191c', linestyle='--', alpha=0.8,
                        label=fr'Optimal $\lambda$ = {self.best_lambda_:.4e}')
            plt.scatter(self.best_lambda_, best_score, color='#d7191c', s=100, zorder=5)

        plt.xlabel(fr"Regularization Parameter ($\lambda$)", fontsize=12)
        plt.ylabel(display_name, fontsize=12)
        plt.title(fr"{display_name} vs $\lambda$", fontsize=17, pad=15)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(frameon=True, loc='best', fontsize=10)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()


    def plot_coefficients(self):
        """
        Produces plot showing the coefficient values as function of lambda parameter.
        """
  
        lams = sorted(self.coef_paths_.keys())
        coefs = np.array([self.coef_paths_[l] for l in lams])
        
        plt.figure(figsize=(12, 7))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        plt.semilogx(lams, coefs, linewidth=1.5, alpha=0.7)

        if self.best_lambda_ is not None:
            plt.axvline(x=self.best_lambda_, color='#d7191c', linestyle='--', 
                        linewidth=2, label=fr'Optimal $\lambda$ ({self.best_lambda_:.2e})')
        
        plt.xlabel(r"Regularization Parameter ($\lambda$)", fontsize=12)
        plt.ylabel(r"Coefficient Values ($\beta$)", fontsize=12)
        plt.title("Lasso Path: Coefficients Shrinking to Zero", fontsize=14)
        
        plt.text(0.02, 0.95, f"Total predictors: {coefs.shape[1]}", 
                transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5))
        
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(loc='upper right', frameon=True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()



def compute_metric(y_true, proba, measure):
    y_pred = (proba >= 0.5).astype(int)

    if measure == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    elif measure == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    elif measure == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    elif measure == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    elif measure == "roc_auc":
        return roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan
    elif measure == "avg_precision":
        return average_precision_score(y_true, proba)
    else:
        raise ValueError(f"Unsupported measure: {measure}")
    
def compute_all_metrics(y_true, proba):
    y_pred = (proba >= 0.5).astype(int)

    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        "avg_precision": average_precision_score(y_true, proba),
    }

def compare_fista_with_sklearn(
        X_train,
        y_train,
        X_test,
        y_test,
        lambdas,
        dataframe,
        measure="roc_auc",
        max_iter=1000, 
        stop_condition=1e-6,
):
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    lambdas = np.array(lambdas)

    fista = LogisticLassoRegressionFISTA(
        lambdas=lambdas,
        measure=measure,
        max_iter=max_iter,
        stop_condition=stop_condition,
    )
    fista.fit(X_train=X_train_s, y_train=y_train)
    fista_best_score = fista.validate(X_valid=X_test_s, y_valid=y_test)
    
    fista_test_proba = fista.predict_proba(X_test=X_test_s)
    fista_all_metrics = compute_all_metrics(y_true=y_test, proba=fista_test_proba)

    sklearn_scores = {}
    sklearn_models = {}

    for lam in lambdas:
        C = 1.0 / lam

        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            fit_intercept=True,
            max_iter=max_iter,
            tol=stop_condition,
            random_state=42,
        )

        clf.fit(X=X_train_s, y=y_train)

        proba_valid = clf.predict_proba(X=X_test_s)[:, 1]
        val_score = compute_metric(y_true=np.array(y_test), proba=proba_valid, measure=measure)

        sklearn_scores[lam] = val_score
        sklearn_models[lam] = clf

    sklearn_best_lambda = max(sklearn_scores, key=sklearn_scores.get)
    sklearn_best_model = sklearn_models[sklearn_best_lambda]
    sklearn_best_score = sklearn_scores[sklearn_best_lambda]

    sklearn_test_proba = sklearn_best_model.predict_proba(X_test_s)[:, 1]
    sklearn_all_metrics = compute_all_metrics(y_test, sklearn_test_proba)

    fista_coef = fista.beta_
    fista_intercept = fista.intercept_

    sklearn_coef = sklearn_best_model.coef_.ravel()
    sklearn_intercept = sklearn_best_model.intercept_[0]

    comparison = {
        "dataframe": dataframe,

        "measure_used_for_lambda_selection": measure,

        "fista_best_lambda": fista.best_lambda_,
        "fista_best_score_on_test": fista_best_score,
        "fista_nonzero_coef": int(np.sum(np.abs(fista_coef) > 1e-8)),
        "fista_intercept": float(fista_intercept),

        "sklearn_best_lambda_equiv": sklearn_best_lambda,
        "sklearn_best_C": float(1.0 / sklearn_best_lambda),
        "sklearn_best_score_on_test": sklearn_best_score,
        "sklearn_nonzero_coef": int(np.sum(np.abs(sklearn_coef) > 1e-8)),
        "sklearn_intercept": float(sklearn_intercept),

        "l2_distance_between_coefs": float(np.linalg.norm(fista_coef - sklearn_coef)),
        "max_abs_difference_between_coefs": float(np.max(np.abs(fista_coef - sklearn_coef))),
    }

    metrics_df = pd.DataFrame([
        {
            "dataframe": dataframe,
            "model": "FISTA",
            "recall": fista_all_metrics["recall"],
            "precision": fista_all_metrics["precision"],
            "f1": fista_all_metrics["f1"],
            "balanced_accuracy": fista_all_metrics["balanced_accuracy"],
            "roc_auc": fista_all_metrics["roc_auc"],
            "avg_precision": fista_all_metrics["avg_precision"],
        },
        {
            "dataframe": dataframe,
            "model": "sklearn",
            "recall": sklearn_all_metrics["recall"],
            "precision": sklearn_all_metrics["precision"],
            "f1": sklearn_all_metrics["f1"],
            "balanced_accuracy": sklearn_all_metrics["balanced_accuracy"],
            "roc_auc": sklearn_all_metrics["roc_auc"],
            "avg_precision": sklearn_all_metrics["avg_precision"],
        }
    ])

    comparison_df = pd.DataFrame([comparison])

    return comparison_df, metrics_df, fista, sklearn_best_model

