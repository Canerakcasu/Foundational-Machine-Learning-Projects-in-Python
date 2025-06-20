# 📘 Machine Learning Fundamentals Practice

This repository contains a collection of four fundamental machine learning projects, each focusing on a different algorithm and task. The goal of these projects is to implement and analyze core ML models using Python and the `scikit-learn` library.

---

## 📁 Project Structure

```
├── task1_linear_regression/
│   ├── main.py
│   └── LifeExpectancy.csv
│
├── task2_decision_tree/
│   ├── main.py
│   └── (Data is loaded from sklearn)
│
├── task3_clustering/
│   ├── main.py
│   ├── s1.txt
│   ├── s2.txt
│   ├── s3.txt
│   ├── s4.txt
│   └── spiral.txt
│
└── task4_naive_bayes/
    ├── main.py
    └── (Data is loaded from sklearn)
```


## 🚀 How to Run

### Prerequisites
Ensure you have **Python** installed.

### Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib
```
Run a Task
Navigate into a specific task directory and execute the script.

# Example for Task 1
cd task1_linear_regression
python main.py



📝 Project Summaries



## Task 1: Linear Regression – Predicting Life Expectancy

**Objective:** Predict life expectancy using both simple and multiple linear regression.

**Dataset:** `LifeExpectancy.csv` (sourced from WHO data).

**Methodology:**
- Load, clean, and split the dataset.
- Perform exploratory data analysis to identify key features.
- Apply Simple Linear Regression using individual features (e.g., GDP, Total expenditure, Alcohol).
- Apply Multiple Linear Regression using the top 4 most correlated features.
- Evaluate model performance using R² and MAE metrics.

**Conclusion:**  
Multiple linear regression (MAE ≈ 3.4–4.2) provides significantly better predictions than single-feature models.

## Task 2: Decision Tree - Iris Flower Classification

**Objective:** Classify Iris species using petal and sepal measurements.

**Dataset:** Iris dataset (loaded from `sklearn.datasets`).

**Methodology:**
- Train a `DecisionTreeClassifier` on the dataset.
- Evaluate accuracy on both training and test splits.
- Experiment with different `random_state` values and split ratios.
- Visualize the resulting decision tree.

**Conclusion:**  
The decision tree achieves high accuracy on clean data, but may overfit the training set. Visualization helps interpret feature importance and decision boundaries.
Objective: Classify Iris species using petal and sepal data.

Dataset: Loaded from sklearn.datasets.

## Task 3: K-Means Clustering – Grouping Unlabeled Data

**Objective:** Apply K-Means to various 2D datasets to explore clustering performance.

**Datasets:** `s1.txt`, `s2.txt`, `s3.txt`, `s4.txt`, `spiral.txt`.

**Methodology:**
- Parse and preprocess noisy and structured text data.
- Apply K-Means clustering and visualize results.

**Analysis:**
- ✅ Spherical clusters (`s1`, `s2`): K-Means performs very well.
- 🔶 Noisy clusters (`s3`): Robust clustering despite noise.
- ❌ Elongated or spiral clusters (`s4`, `spiral`): K-Means struggles with non-globular shapes.

**Conclusion:**  
K-Means is effective for globular clusters but not suitable for complex or non-convex patterns.

---

## Task 4: Naive Bayes – Handwritten Digit Recognition

**Objective:** Recognize handwritten digits (0–9) using Gaussian Naive Bayes.

**Dataset:** MNIST (loaded via `fetch_openml`).

**Methodology:**
- Load and preprocess image data.
- Classify using `GaussianNB` (handles continuous pixel values).
- Evaluate with accuracy, classification report, and confusion matrix.

**Conclusion:**  
Achieves around 55% accuracy. Most errors occur between visually similar digits (e.g., 4 vs. 9).

📌 Summary
This repository covers:

🔬 Supervised learning: Regression, Classification

📊 Unsupervised learning: Clustering

🧠 Key concepts: Overfitting, evaluation metrics, data visualization, model limitations

