import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

method_palette = {
    'Oracle': '#333333',
    'Naive': '#d73027',
    'Unlabeled-EM': '#4575b4',
    'Unlabeled-KNN': '#74add1'
}

def load_and_prepare_data(folder_path="results/"):
    files = glob.glob(os.path.join(folder_path, "*_results.csv"))
    all_dfs = []
    
    for f in files:
        temp_df = pd.read_csv(f, index_col=0)
        all_dfs.append(temp_df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    method_map = {
        'Oracle': 'Oracle',
        'Naive': 'Naive',
        'KNN_hard': 'Unlabeled-KNN-hard',
        'KNN_proba': 'Unlabeled-KNN-proba',
        'EM': 'Unlabeled-EM'
    }
    full_df['method'] = full_df['method'].apply(lambda x :method_map.get(x, x))
    
    return full_df

def get_aggregated_results(df):
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc']
    
    agg_df = df.groupby(['dataset', 'method', 'missing_schema', 'missing_rate'])[metrics].agg(['mean', 'std']).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_df.columns]
    
    return agg_df