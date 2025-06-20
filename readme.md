# 📘 Machine Learning Fundamentals Practice

This repository contains a collection of four fundamental machine learning projects, each focusing on a different algorithm and task. The goal of these projects is to implement and analyze core ML models using Python and the `scikit-learn` library.

---

## 📁 Project Structure

.
├── task1_linear_regression/
│ ├── main.py
│ └── LifeExpectancy.csv
│
├── task2_decision_tree/
│ ├── main.py
│ └── (Data is loaded from sklearn)
│
├── task3_clustering/
│ ├── main.py
│ ├── s1.txt
│ ├── s2.txt
│ ├── s3.txt
│ ├── s4.txt
│ └── spiral.txt
│
└── task4_naive_bayes/
├── main.py
└── (Data is loaded from sklearn)


---

## 🚀 How to Run

### Prerequisites
Ensure you have **Python** installed.

### Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib

Run a Task
Navigate into a specific task directory and execute the script.

# Example for Task 1
cd task1_linear_regression
python main.py



📝 Project Summaries
🔹 Task 1: Linear Regression - Predicting Life Expectancy
Objective: Predict life expectancy using simple and multiple linear regression.

Dataset: LifeExpectancy.csv (WHO data).

Methodology:

Data loading, cleaning, and splitting.

Exploratory data analysis.

Simple Linear Regression with individual features: GDP, Total expenditure, Alcohol.

Multiple Linear Regression with top 4 correlated features.

Evaluation using R² and MAE.

Conclusion:
Multiple linear regression (MAE ≈ 3.4–4.2) significantly outperforms single-feature models.

🔹 Task 2: Decision Tree - Iris Flower Classification
Objective: Classify Iris species using petal and sepal data.

Dataset: Loaded from sklearn.datasets.

Methodology:

Training with DecisionTreeClassifier.

Evaluating accuracy on train/test splits.

Exploring effects of random_state and split ratios.

Visualizing the decision tree.

Conclusion:
Demonstrates high performance on clean data and highlights overfitting on training set.

🔹 Task 3: K-Means Clustering - Grouping Unlabeled Data
Objective: Apply K-Means on varied 2D datasets to evaluate clustering behavior.

Datasets: s1.txt, s2.txt, s3.txt, s4.txt, spiral.txt.

Methodology:

Custom parser for noisy and structured text data.

K-Means applied and visualized.

Analysis:

✅ Spherical clusters (s1, s2) – excellent performance.

🔶 Noisy clusters (s3) – robust clustering.

❌ Elongated/spiral clusters (s4, spiral) – K-Means struggles.

Conclusion:
K-Means is ideal for globular clusters, not complex or non-convex shapes.

🔹 Task 4: Naive Bayes - Handwritten Digit Recognition
Objective: Recognize digits (0–9) using Gaussian Naive Bayes.

Dataset: MNIST (loaded via fetch_openml).

Methodology:

Data loading and preprocessing.

Classification using GaussianNB (suitable for continuous pixel values).

Evaluation via accuracy, classification report, and confusion matrix.

Conclusion:
Achieves ~55% accuracy. Most confusion occurs between visually similar digits (e.g., 4 vs. 9).

📌 Summary
This repository covers:

🔬 Supervised learning: Regression, Classification

📊 Unsupervised learning: Clustering

🧠 Key concepts: Overfitting, evaluation metrics, data visualization, model limitations

