# Knn-Fraud-Detection


This notebook aims to build a machine learning model to detect payment fraud. The process involves loading the payment fraud dataset, exploring its structure and identifying missing values, and then preprocessing the data by handling missing values and encoding categorical features. Finally, a K-Nearest Neighbors (KNN) classification model is trained on the processed data, and its performance is evaluated.

## Dataset Description

The dataset used in this notebook is related to payment fraud detection. It contains information about online transactions, including details about the account, the items purchased, the payment method, and whether the transaction was fraudulent or not.

The dataset has 39,221 rows and 8 columns. The columns are:
- `accountAgeDays`: Age of the account in days (integer).
- `numItems`: Number of items purchased in the transaction (integer).
- `localTime`: Local time of the transaction (float).
- `paymentMethod`: Payment method used for the transaction (categorical - object, later encoded).
- `paymentMethodAgeDays`: Age of the payment method in days (float).
- `Category`: Category of the purchased items (categorical - object, later encoded).
- `isWeekend`: Whether the transaction occurred on a weekend (float).
- `label`: The target variable, indicating whether the transaction is fraudulent (1) or not (0) (integer).

The target variable `label` is crucial for the fraud detection task, as the goal is to build a model that can accurately predict this variable.

## Data Cleaning and Preprocessing

This section details the steps taken to clean and preprocess the dataset before training the machine learning model.

1.  **Handling Missing Values:**
    -   The `isWeekend` column had missing values. These were imputed using the median strategy with `sklearn.impute.SimpleImputer`. The median was chosen to be robust to potential outliers.
    -   The `Category` column also contained missing values. These were filled with the most frequent category (mode) in the column using the `fillna()` method combined with `mode()[0]`. This is a suitable strategy for categorical data.

2.  **Encoding Categorical Features:**
    -   Machine learning models require numerical input, so the categorical features `paymentMethod` and `Category` were converted into numerical format using `sklearn.preprocessing.LabelEncoder`.
    -   `LabelEncoder` assigns a unique integer to each unique category within a column. For example, 'creditcard', 'paypal', and 'storecredit' in the `paymentMethod` column would be mapped to distinct integers.

These steps ensure that the dataset is in a suitable format for training the classification model.

## Model Training

This section outlines the steps involved in training the machine learning model for payment fraud detection.

1.  **Data Splitting:**
    -   The dataset was split into features (`X`) and the target variable (`y`, which is the 'label' column).
    -   The data was then divided into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.
    -   A `test_size` of 20% was used, meaning 80% of the data was allocated for training and 20% for testing.
    -   `random_state=42` was set to ensure reproducibility of the split.
    -   `stratify=y` was used to ensure that the proportion of fraudulent (label=1) and non-fraudulent (label=0) transactions was maintained in both the training and testing sets. This is particularly important for imbalanced datasets like this one.

2.  **Feature Scaling:**
    -   Before training the model, the features (`X`) were scaled using `StandardScaler` from `sklearn.preprocessing`.
    -   Scaling is crucial for algorithms like KNN that are sensitive to the magnitude of features. `StandardScaler` standardizes features by removing the mean and scaling to unit variance.
    -   The scaler was fitted on the training data (`X_train`) and then used to transform both the training (`X_train_scaled`) and testing (`X_test_scaled`) sets.

3.  **Model Selection and Training:**
    -   A K-Nearest Neighbors (KNN) classifier from `sklearn.neighbors` was chosen for this task.
    -   The KNN model was initialized with the following parameters:
        -   `n_neighbors=5`: The number of neighbors to consider when making a prediction.
        -   `weights='uniform'`: All points in the neighborhood are weighted equally.
        -   `metric='manhattan'`: The Manhattan distance (L1 norm) was used as the distance metric.
    -   The KNN model was then trained (fitted) on the scaled training data (`X_train_scaled`) and the corresponding training labels (`y_train`).

  ## Model Performance and Results

This section summarizes the performance of the trained K-Nearest Neighbors (KNN) classifier on the test dataset.

The model achieved an overall **accuracy of 0.9782**.

### Classification Report

The classification report provides detailed metrics for each class (0: non-fraudulent, 1: fraudulent):

| Metric    | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.99      | 0.98   | 0.99     | 7733    |
| Class 1   | 0.35      | 0.62   | 0.45     | 112     |
| **Macro Avg** | 0.67      | 0.80   | 0.72     | 7845    |
| **Weighted Avg** | 0.99      | 0.98   | 0.98     | 7845    |

-   **Precision:** For non-fraudulent transactions (Class 0), the precision is 0.99, meaning that when the model predicts a transaction is non-fraudulent, it is correct 99% of the time. For fraudulent transactions (Class 1), the precision is 0.35, indicating that when the model predicts a transaction is fraudulent, it is correct only 35% of the time.
-   **Recall:** For non-fraudulent transactions (Class 0), the recall is 0.98, meaning the model correctly identifies 98% of all actual non-fraudulent transactions. For fraudulent transactions (Class 1), the recall is 0.62, meaning the model correctly identifies 62% of all actual fraudulent transactions.
-   **F1-score:** The F1-score is the harmonic mean of precision and recall. It provides a balance between the two metrics. The F1-score is high for Class 0 (0.99) and lower for Class 1 (0.45).
-   **Support:** The number of actual instances in each class in the test set. There are 7733 non-fraudulent and 112 fraudulent transactions.

### Confusion Matrix

The confusion matrix provides a breakdown of the model's predictions versus the actual labels:

[[7604  129]
 [  42   70]]

-   **True Positives (TP):** 70 (The model correctly predicted 70 fraudulent transactions).
-   **True Negatives (TN):** 7604 (The model correctly predicted 7604 non-fraudulent transactions).
-   **False Positives (FP):** 129 (The model incorrectly predicted 129 non-fraudulent transactions as fraudulent - Type I error).
-   **False Negatives (FN):** 42 (The model incorrectly predicted 42 fraudulent transactions as non-fraudulent - Type II error).


