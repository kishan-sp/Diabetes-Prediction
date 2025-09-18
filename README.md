# Diabetes Prediction Project
Project Overview
This project aims to predict the onset of diabetes based on a set of diagnostic medical attributes. Using the Pima Indians Diabetes Dataset, this analysis involves data exploration, preprocessing, and the implementation of machine learning models to classify patients as diabetic or non-diabetic. The primary models used for prediction are Logistic Regression and K-Nearest Neighbors (KNN). The evaluation focuses on metrics suitable for an imbalanced dataset, prioritizing recall for the positive class (diabetic) to minimize false negatives.

## Dataset
The dataset used is the Pima Indians Diabetes Database, which is a well-known dataset from the UCI Machine Learning Repository. It contains the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: A function that scores the likelihood of diabetes based on family history.
- Age: Age in years
- Outcome: Class variable (0 for non-diabetic, 1 for diabetic)

## Installation
To run the analysis, you will need Python and the following libraries installed. You can install them using pip:
```
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Usage
The primary analysis is contained within the Diabetes Prediction.ipynb Jupyter Notebook. To view or run the analysis:

1. Clone or download this repository.
2. Ensure you have Jupyter Notebook or JupyterLab installed.
3. Navigate to the project directory and launch Jupyter:
   ``` jupyter notebook "Diabetes Prediction.ipynb" ```
5. You can then execute the cells sequentially to reproduce the analysis.

## Exploratory Data Analysis (EDA)
- The EDA phase revealed several key insights:
- The dataset is imbalanced, with significantly more non-diabetic cases than diabetic ones.
- Several features, such as Glucose, BloodPressure, SkinThickness, Insulin, and BMI, had zero values, which are physiologically impossible and were treated as missing data. These were imputed using the mean or median of the respective columns.
- Visualizations like histograms and boxplots showed the distribution of each feature and the presence of outliers.
- A correlation heatmap was generated to understand the relationships between different variables.

## Modeling and Evaluation
**Two machine learning models were trained and evaluated:**

1. Logistic Regression: A linear model used for binary classification.
2. K-Nearest Neighbors (KNN): A non-parametric algorithm that classifies data points based on the majority class of their nearest neighbors.
The data was split into training (80%) and testing (20%) sets. The features were scaled using MinMaxScaler to ensure that all variables contributed equally to the model's performance.

The models were evaluated based on the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

## Results
The performance of the models on the test set is summarized below.

**Logistic Regression:**
- Recall (for diabetic class): 0.44
- Macro Avg F1-Score: 0.65
- Weighted Avg F1-Score: 0.69

**K-Nearest Neighbors (KNN):**
- Recall (for diabetic class): 0.52
- Macro Avg F1-Score: 0.67
- Weighted Avg F1-Score: 0.70

## Conclusion
- Given the context of medical diagnosis, minimizing false negatives (failing to identify a diabetic patient) is critical. Therefore, Recall for the diabetic class (1) is a primary metric for model selection.
- The KNN model achieved a higher recall (0.52) for the diabetic class compared to Logistic Regression (0.44).
- Considering the F1-score, which balances precision and recall, the KNN model also shows a slightly better macro average F1-score (0.67 vs 0.65).

Based on these results, the K-Nearest Neighbors model is the preferred model for this specific problem, as it is better at correctly identifying patients with diabetes, which is the most crucial outcome to avoid. Further improvements could be explored through hyperparameter tuning, trying more complex models, or employing advanced techniques for handling imbalanced data.
