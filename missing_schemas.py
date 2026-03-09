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
        X, Y_obs = MCAR(X, Y, missing_rate)

    if scheme == "MCAR1":
        raise NotImplementedError("Scheme {scheme} to be implemented.")
    
    if scheme == "MCAR2":
        raise NotImplementedError("Scheme {scheme} to be implemented.")
    
    if scheme == "MNAR":
        raise NotImplementedError("Scheme {scheme} to be implemented.")


    return X, Y_obs

