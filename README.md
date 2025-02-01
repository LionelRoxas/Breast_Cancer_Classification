# Breast Cancer Classification Using Machine Learning

### Project Overview
This project implements and compares three machine learning models (K-Nearest Neighbors, Decision Trees, and Random Forest) for breast cancer classification using the Wisconsin Breast Cancer dataset from sklearn.

### Project Contents
- breast_cancer_classification.ipynb: Jupyter notebook containing all code and analysis
- README.md: Project documentation and overview
- Project_Report.pdf: Detailed analysis and findings

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
Key Features
```

### Implementation of three classification models:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest


### Model performance evaluation
- Hyperparameter tuning analysis
- Feature importance visualization

### Results
- Best performing model (Random Forest):
  - Accuracy: 96.49%
  - Precision: 95.89%
  - Recall: 98.59%
  - F1-score: 97.22%

### Key Findings
Random Forest performed best across all metrics

Top 2 most important features for cancer detection:
- Worst concave points
- Worst area

Model hyperparameter optimization showed:
- KNN: optimal with 10 neighbors
- Decision Tree: stable after depth of 3
- Random Forest: effective with 50-100 trees



### Usage
1. Clone the repository
2. Install required packages using pip:
```pip install numpy pandas matplotlib seaborn scikit-learn```

3. Open and run the Jupyter notebook:
```jupyter notebook breast_cancer_classification.ipynb```

### Author:

[Lionel Roxas](https://github.com/LionelRoxas/)
