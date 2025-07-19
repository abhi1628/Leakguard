# LeakGuard

**LeakGuard** is a lightweight, modular Python tool designed to detect and quantify data leakage in machine learning workflows. It includes functions to identify common leakage patterns such as preprocessing leakage and target leakage, and computes the Leakage Impact Score (LIS) to assess their effect on model performance.

## 🔧 Features

- Detects preprocessing leakage (e.g., scaling before train-test split)
- Detects feature-target leakage via correlation analysis
- Computes Leakage Impact Score (LIS)
- Simple integration into scikit-learn pipelines

## 📦 Installation

```bash
pip install pandas numpy scikit-learn
```

## 🚀 Usage Example

```python
from leakguard import LeakGuard
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Initialize LeakGuard
lg = LeakGuard()

# Simulate preprocessing leakage
lg.detect_preprocessing_leakage()

# Simulate target leakage detection
lg.detect_target_leakage(X, y)

# Compare performance and compute LIS
lis = lg.compute_lis(accuracy_with_leak, accuracy_baseline)
print("Leakage Impact Score:", lis)
```

## 📄 License

This tool is released under the MIT License.

## ✍️ Author

Developed by [Abhishek Singh]. For academic and research use.
