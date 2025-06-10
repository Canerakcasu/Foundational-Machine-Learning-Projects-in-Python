Machine Learning Fundamentals Practice
This repository contains a collection of four fundamental machine learning projects, each focusing on a different algorithm and task. The goal of these projects is to implement and analyze core ML models using Python and the scikit-learn library.

Each task is contained within its own folder and includes the necessary Python script (main.py) and data files.

Project Structure
.
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

How to Run
Prerequisites: Ensure you have Python installed.

Dependencies: Install the required libraries using pip.

pip install scikit-learn pandas numpy matplotlib

Execution: Navigate into a specific task's directory and run the main script.

cd task1_linear_regression
python main.py

Task 1: Linear Regression - Predicting Life Expectancy
Objective: To predict life expectancy based on various health and economic factors using simple and multiple linear regression.

Dataset: LifeExpectancy.csv, containing data from the World Health Organization.

Methodology:

Data Preparation: The dataset was loaded, cleaned, and split into training and test sets.

Exploratory Analysis: A preliminary analysis was conducted to understand the data distribution.

Simple Linear Regression: Three separate models were trained to predict life expectancy using 'GDP', 'Total expenditure', and 'Alcohol' as single features.

Multiple Linear Regression: A model was trained using the four features with the highest correlation to the target variable.

Evaluation: Models were compared based on their R-squared scores and Mean Absolute Error (MAE) on the test set.

Conclusion: The multiple linear regression model (MAE ≈ 3.4-4.2) significantly outperformed all simple regression models, demonstrating that a combination of relevant factors provides a much more accurate prediction of life expectancy.

Task 2: Decision Tree - Iris Flower Classification
Objective: To classify Iris flower species (setosa, versicolor, virginica) based on sepal and petal measurements.

Dataset: The classic Iris dataset, loaded directly from scikit-learn.

Methodology:

Model Training: A DecisionTreeClassifier was trained on the Iris dataset.

Evaluation: Model accuracy was calculated for both the training and test sets. The 100% accuracy on the training data highlighted the concept of overfitting.

Parameter Analysis: The script analyzes how model performance changes with different random_state values and various train_test_split ratios.

Visualization: The final, trained decision tree was plotted to visually represent its classification logic.

Conclusion: This exercise demonstrates the implementation of a decision tree classifier, showing its high performance on separable data while also illustrating its sensitivity to training data variations.

Task 3: K-Means Clustering - Grouping Unlabeled Data
Objective: To apply the K-Means algorithm to various datasets to understand its strengths and weaknesses based on data structure.

Dataset: Five .txt files (s1-s4, spiral) containing 2D data points with different structural properties.

Methodology:

Parser: A robust data parser was created to handle all provided text files, including those with corrupted or non-numeric lines.

K-Means Application: The algorithm was applied to each dataset.

Analysis: The results were visualized, comparing the algorithm's performance on:

Globular, well-separated clusters (s1, s2): K-Means performed exceptionally well.

Noisy clusters (s3): K-Means was robust and identified the underlying clusters successfully.

Non-globular clusters (s4): K-Means failed to capture the elongated cluster structures.

Spiral Dataset Test: K-Means was applied to the spiral.txt dataset and the result was compared against the ground truth.

Conclusion: The project effectively demonstrates that K-Means is a powerful and efficient algorithm for spherical clusters but is unsuitable for more complex, non-globular structures. This highlights the importance of choosing a clustering algorithm that matches the underlying geometry of the data.

Task 4: Naive Bayes - Handwritten Digit Recognition
Objective: To build a classifier to recognize handwritten digits (0-9) using the MNIST dataset.

Dataset: The MNIST dataset, containing 70,000 images of 28x28 pixels each.

Methodology:

Data Handling: The data was loaded via sklearn.datasets.fetch_openml, and its characteristics were logged to a file.

Model Selection: GaussianNB was chosen as the appropriate model because the pixel values are continuous (0-255), fitting the Gaussian distribution assumption.

Evaluation: The model's performance was evaluated using:

Overall accuracy score.

A detailed classification report showing precision, recall, and F1-score for each digit.

A confusion matrix to visually analyze which digits were most often confused with each other (e.g., 4 vs. 9).

Conclusion: Despite its simplicity and high speed, Gaussian Naive Bayes achieves a respectable baseline accuracy (around 55%). The analysis of the confusion matrix confirms that most errors occur between visually similar digits, providing clear insight into the model's limitations.
