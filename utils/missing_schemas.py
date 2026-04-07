import pandas as pd
import numpy as np

def MCAR(X, Y, missing_rate):

    """
    Function implementing the Missing Completely At Random (MCAR) schema in Y.

    Parameters:
        X: pd.DataFrame - explanatory variables
        Y: pd.Series - target variable
        missing_rate: float - fraction of missing Y

    Returns:
        X: pd.DataFrame (unchanged)
        Y_obs: pd.Series - Y with -1 where Y is missing
    """

    Y_obs = Y.copy()
    n=len(Y)
    n_missing= int(n*missing_rate)

    #random choice for indices of Y which will be marked as missing
    missing_indices = np.random.choice(a=n, size=n_missing, replace=False)
    Y_obs.iloc[missing_indices] = -1

    return X, Y_obs

def MAR1(X, Y, missing_rate):

    """
    Function implementing the Missing At Random (MAR1) schema in Y - assumes that the missing data mechanism depends only on a single explanatory variable.
    In this implementation the "single explanatory variable" is always the first column.

    Parameters:
        X: pd.DataFrame - explanatory variables
        Y: pd.Series - target variable
        missing_rate: float - fraction of missing Y

    Returns:
        X: pd.DataFrame (unchanged)
        Y_obs: pd.Series - Y with -1 where Y is missing
    """

    Y_obs = Y.copy()
    n = len(Y)
    n_missing= int(n * missing_rate)

    feature = X.iloc[:, 0]
    weights = feature - feature.min() + 1e-6
    probabilities = weights / weights.sum()   # the bigger the value the higher probability it is going to be changed to missing - dependence on one feature

    missing_indices = np.random.choice(a=n, size=n_missing, replace=False, p=probabilities)
    Y_obs.iloc[missing_indices] = -1

    return X, Y_obs

def MAR2(X, Y, missing_rate):

    """
    Function implementing the Missing At Random (MAR2) schema in Y - assumes that the missing data mechanism depends on all explanatory variables.

    Parameters:
        X: pd.DataFrame - explanatory variables
        Y: pd.Series - target variable
        missing_rate: float - fraction of missing Y

    Returns:
        X: pd.DataFrame (unchanged)
        Y_obs: pd.Series - Y with -1 where Y is missing
    """

    Y_obs = Y.copy()
    n = len(Y)
    n_missing = int(n * missing_rate)

    X_array = np.array(X)
    X_standardized = (X_array - X_array.mean(axis=0)) / (X_array.std(axis=0) + 1e-6)  # standarization (bcs if there is one column with high values it would have bigger influance than the one with smaller values)
    row_score = X_standardized.sum(axis=1)
    weights = row_score - row_score.min() + 1e-6
    probabilities = weights / weights.sum()

    missing_indices = np.random.choice(a=n, size=n_missing, replace=False, p=probabilities)
    Y_obs.iloc[missing_indices] = -1

    return X, Y_obs

def MNAR(X, Y, missing_rate):

    """
    Function implementing the Missing Not At Random (MNAR) schema in Y - assumes that the missing data mechanism depends on both explanatory variables and target variable.
    Adapting the MAR2 schema but adding the influence of value of Y.
    Observations with Y=1 have increased probability of being missing (gamma_1 > 0),
    observations with Y=0 have decreased probability of being missing (gamma_0 < 0).

    Parameters:
        X: pd.DataFrame - explanatory variables
        Y: pd.Series - target variable
        missing_rate: float - fraction of missing Y

    Returns:
        X: pd.DataFrame (unchanged)
        Y_obs: pd.Series - Y with -1 where Y is missing
    """

    Y_obs = Y.copy()
    n = len(Y)
    n_missing = int(n * missing_rate)

    gamma_0 = -0.5 #added score when Y=0 (negative = less likely missing)
    gamma_1 = 2.0 #added score when Y=1 (positive = more likely missing)

    #the same as in the MAR2 schema
    X_array = np.array(X)
    X_standardized = (X_array - X_array.mean(axis=0)) / (X_array.std(axis=0) + 1e-6)  # standarization (bcs if there is one column with high values it would have bigger influance than the one with smaller values)
    x_score = X_standardized.sum(axis=1)

    #asymmetrical influence from both classes in Y
    y_score = np.where(np.array(Y) == 1, gamma_1, gamma_0)

    combined_score = x_score+y_score

    weights = combined_score - combined_score.min() + 1e-6
    probabilities = weights / weights.sum()

    missing_indices = np.random.choice(a=n, size=n_missing, replace=False, p=probabilities)
    Y_obs.iloc[missing_indices] = -1

    return X, Y_obs

def generate_missing_y(X, Y, scheme, missing_rate):

    """
    Generate missing Y based on the chosen scheme.
    
    Parameters:
        X: pd.DataFrame - explanatory variables
        Y: pd.Series - target variable
        scheme: str - 'MCAR', 'MAR1', 'MAR2', 'MNAR'
        missing_rate: float - fraction of missing Y
    
    Returns:
        X: pd.DataFrame (unchanged)
        Y_obs: pd.Series - Y with -1 where Y is missing
    """

    schemes = ['MCAR', 'MAR1', 'MAR2', 'MNAR']
    if scheme not in schemes:
        raise ValueError(f"Invalid scheme '{scheme}'. Choose one of {schemes}.")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(Y, pd.Series):
        raise TypeError("Y must be a pandas Series.")
    if not (0 <= missing_rate <= 1):
        raise ValueError("missing_rate must be a float between 0 and 1.")

    if scheme == "MCAR":
        X, Y_obs = MCAR(X=X, Y=Y, missing_rate=missing_rate)

    elif scheme == "MAR1":
        X, Y_obs = MAR1(X=X, Y=Y, missing_rate=missing_rate)
    
    elif scheme == "MAR2":
        X, Y_obs = MAR2(X=X, Y=Y, missing_rate=missing_rate)
    
    elif scheme == "MNAR":
        X, Y_obs = MNAR(X=X, Y=Y, missing_rate=missing_rate)


    return X, Y_obs

