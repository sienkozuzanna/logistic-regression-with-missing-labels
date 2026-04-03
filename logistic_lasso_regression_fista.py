import numpy as np

class LogisticLassoRegressionFISTA:
    def __init__(self, lambda_=1.0, max_iter=1000, stop_condition=1e-6, step_size=None):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.stop_condition = stop_condition
        self.step_size = step_size

        self.beta_ = None
        self.intercept_ = None
        self.objective_history_ = []
        self.n_iter_ = 0

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
        spectral_norm = np.linalg.norm(X, 2)
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
    
    def _objective(self, X, y):
        """
        Calculates full objective function:
            logistic_loss + lambda * |beta|

        Parameters:
            X: pd.DataFrame - explanatory variables
            y: pd.Series - target variable

        Returns:
            float - value of the full objective function
        """
        loss = self._compute_logistic_loss(X=X, y=y, beta=self.beta_, intercept=self.intercept_)
        L1_penalty = self.lambda_ * np.sum(np.abs(self.beta_))
        return loss + L1_penalty

    def fit(self, X_train, y_train):
        X = np.array(X_train)
        y = np.array(y_train)

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

            self.beta_ = beta.copy()
            self.intercept_ = intercept
            self.objective_history_.append(self._objective(X=X, y=y))

            param_change = np.linalg.norm(beta - beta_prev) + abs(intercept - intercept_prev)
            if param_change < self.stop_condition:
                self.n_iter_ = i + 1
                self.beta_ = beta
                self.intercept_ = intercept
                return self
        
        self.n_iter_ = self.max_iter
        self.beta_ = beta
        self.intercept_ = intercept

        return self

    def predict_proba(self, X_test):
        """
        Returns predicted probabilities for class 1.
        """
        X = np.array(X_test)
        scores = X @ self.beta_ + self.intercept_
        return self._sigmoid(scores)