# Advanced Machine Learning - project 1: Logistic Regression with Missing Labels

The goal of this project is to investigate logistic regression with missing labels in the training set.
The project consists of three main parts:
1. Data preparation and simulation of missing-label mechanisms (MCAR, MAR1, MAR2, MNAR).
2. Custom implementation of L1-regularized logistic regression optimized with FISTA.
3. Semi-supervised techniques for handling missing labels and comparison with naive and oracle methods.

## Repository structure
- `utils/missing_schemes.py` - generation of MCAR, MAR1, MAR2 and MNAR missing labels
- `utils/logistic_lasso_regression_fista.py` - custom FISTA implementation
- `utils/unlabled_log_reg.py` - semi-supervised, naive and oracle methods
- `notebooks/` - demonstration notebooks and experiments
- `data/` - data used in experiments (kept in `.gitignore` and not stored in the repository - can be loaded from `notebooks/fetch_data.ipynb`)
- `README.md` - project description and usage instruction

## Instructions to run the FISTA Logistic Regression with L1 penalty
This section describes how to run the custom implmenetation of L1-regularized logistic regression optimized with FISTA algorithm.

### 1. Prepare the data
The input data should consist of:
- `X` - feature matrix (numerical data)
- `y` - binary labels (0 or 1)

Split the data into training and test sets, e.g. using `train_test_split` from `sklearn` library.

Scale the data, e.g. with `MinMaxScaler` from `sklearn` library.

### 2. Initialize the model
Import the model class with: `from utils.logistic_lasso_regression_fista import LogisticLassoRegressionFISTA`
and initialize the model:

```python
model = LogisticLassoRegressionFISTA(
    lambdas=None,        # default logarithmic grid   
    measure="roc_auc",   # metric used for lambda selection    
    max_iter=1000,    
    stop_condition=1e-6,    
)
```

### 3. Fit the model
Fit the model by running: `model.fit(X_train, y_train)`

This step fits the model for all candidate values of the regularization parameter &lambda.

### 4. Select the best lambda
```python
best_score = model.validate(X_test, y_test)

print("Best validation score: ", best_score)
print("Selected lambda: ", model.best_lambda_)
```

The final model is selected only after calling `validate(...)`.

### 5. Make predictions
```python
y_proba = model.predict_proba(X_test)

y_pred = (y_proba >= 0.5).astype(int)
```

### 6. Evaluate performance
Evaluate model's performance using measures such as accuracy and ROC AUC first importing proper libraries from `sklearn` library.

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
```

### 7. Plot results
Optionally you can also plot results for visualizations:
- metric vs lambda 
  `model.plot(X_test, y_test)`
- coefficient paths
  `model.plot_coefficients()`

