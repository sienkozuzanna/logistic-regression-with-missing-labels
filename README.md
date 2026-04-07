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
