
import pandas as pd
import numpy as np

class LeakGuard:
    def __init__(self):
        pass

    def detect_preprocessing_leakage(self):
        print("[LeakGuard] Warning: Preprocessing applied before data split!")

    def detect_target_leakage(self, X, y, threshold=0.8):
        correlations = pd.concat([X, pd.Series(y, name='target')], axis=1).corr()['target'].drop('target')
        high_corr = correlations[abs(correlations) > threshold]
        if not high_corr.empty:
            print("[LeakGuard] Potential target leakage detected in:")
            print(high_corr)
        else:
            print("[LeakGuard] No strong correlation-based target leakage detected.")

    def compute_lis(self, metric_leaky, metric_clean):
        return round(abs(metric_leaky - metric_clean), 4)
