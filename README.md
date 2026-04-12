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
- - `utils/experiment.py` - full experiment runner (sequential and parallel)
- `notebooks/` - demonstration notebooks and experiments
- `data/` - data used in experiments (kept in `.gitignore` and not stored in the repository - can be loaded from `notebooks/fetch_data.ipynb`)
- `requirements.txt` - list of required Python packages
- `README.md` - project description and usage instruction

## Requirements
 
Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Notebooks
 
| Notebook | Description |
|---|---|
| `notebooks/fetch_data.ipynb` | Downloads and saves all datasets to `data/` — **run this first** |
| `notebooks/fista_notebook.ipynb` | FISTA experiments and comparison with sklearn |
| `example_usage.ipynb` | Step-by-step usage guide for all classes |

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

#### Model parameters
The following parameters can be specified:
- `lambdas` (*array-like or None*)
  
  A list or array of regularization parameters (&lambda) to evaluate.
  
  If `None`, a default logarithmic grid is used: &lambda; &isin; [10<sup>-4</sup>, 10<sup>1</sup>]
- `measure` (*str*)
  
  Evaluation metric used to select the best &lambda on the test set.

  Options:
  - `"roc_auc"` (default)
  - `"f1"`
  - `"precision"`
  - `"recall"`
  - `"balanced_accuracy"`
  - `"avg_precision"`
- `max_iter` (*int*)
  
  Maximum number of iterations of the FISTA optimization algorithm.
- `stop_condition` (*float*)
  
  Convergence threshold based on the change in model parameters between iterations.
- `step_size` (*float or None*)
  
  Step size used in gradient updates.
  
  If `None`, it is automatically computed based on an estimate of the Lipschitz constant of the gradient.

### 3. Fit the model
Fit the model by running: 
```python
model.fit(X_train, y_train)
```

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
 
## Instructions to simulate missing labels
 
Import the dispatcher function:
```python
from utils.missing_schemas import generate_missing_y
```

```python
# schema: one of "MCAR", "MAR1", "MAR2", "MNAR"
# missing_rate: fraction of labels to mask, e.g. 0.5 = 50% missing
X_miss, y_miss = generate_missing_y(X_train, y_train, scheme="MCAR", missing_rate=0.5)
# missing labels are indicated by -1 in y_miss
```
 
Available schemas:
- **MCAR** — labels are masked uniformly at random, independently of features and target
- **MAR1** — missingness depends on a single feature (first column)
- **MAR2** — missingness depends on all features
- **MNAR** — missingness depends on both features and the true label value


## Instructions to run semi-supervised methods
 
Import the models:
```python
from utils.unlabeled_log_reg import UnlabeledLogReg, OracleLogReg, NaiveLogReg
```
 
All models are fit with both a training set (with missing labels) and a validation set for lambda selection.
 
**Oracle** — trained on all labels, no missing data (upper bound):
```python
model = OracleLogReg()
model.fit(X_train_full, y_train_full, X_valid, y_valid)
```
 
**Naive** — trained only on labeled samples, unlabeled samples are discarded (lower bound):
```python
model = NaiveLogReg()
model.fit(X_train_miss, y_train_miss, X_valid, y_valid)
```
 
**KNN with hard labels** — missing labels imputed with the predicted class from k-NN:
```python
model = UnlabeledLogReg(method="KNN", label_type="hard", n_neighbors=5)
model.fit(X_train_miss, y_train_miss, X_valid, y_valid)
```
 
**KNN with soft labels** — missing labels imputed with predicted probabilities from k-NN:
```python
model = UnlabeledLogReg(method="KNN", label_type="proba", n_neighbors=5)
model.fit(X_train_miss, y_train_miss, X_valid, y_valid)
```
 
**EM** — iterative Expectation-Maximization over labeled and unlabeled samples:
```python
model = UnlabeledLogReg(method="EM", max_em_iter=10)
model.fit(X_train_miss, y_train_miss, X_valid, y_valid)
```
 
All models share the same evaluation interface:
```python
results = model.evaluate(X_test, y_test, dataset_name="MyDataset", missing_schema="MCAR")
# returns dict with: accuracy, balanced_accuracy, f1_score, roc_auc
```

## Instructions to run the full experiment
 
```python
from experiment import run_full_experiment           # sequential
from experiment import run_full_experiment_parallel  # parallel (recommended for large datasets)
 
results_df = run_full_experiment(X=X, y=y, dataset_name="MyDataset", missing_rates=[0.3, 0.5, 0.8])
```
 
This evaluates all five methods (Oracle, Naive, KNN_hard, KNN_proba, EM) across all combinations of missing rates, schemas (MCAR, MAR1, MAR2, MNAR) and 5 stratified cross-validation folds, returning a `pd.DataFrame` with all results.