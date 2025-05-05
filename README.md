# Engineering Student Placement Prediction using Machine Learning

## Project Overview

This project focuses on predicting whether an engineering student will be placed or not based on academic, demographic, and internship-related features. Using a publicly available dataset, multiple classification algorithms were applied, and Random Forest was chosen as the final model based on its accuracy and interpretability.

## Dataset

Source: Kaggle

Filename: collegePlace.csv

Target Column: PlacedOrNot

Total Records: 2966

Features include: Age, Gender, Stream, CGPA, Internship count, Hostel, History of Backlogs

## Tools and Technologies

Languages: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, joblib

Environment: Jupyter Notebook

## Exploratory Data Analysis (EDA)

Visualized distributions and boxplots to detect outliers (e.g., Age vs Placement)

Analyzed correlations using corr() and heatmaps

Discovered CGPA and Internship count were most correlated with placement

## Data Preprocessing

Handled missing values (if any)

One-Hot Encoding applied to categorical variables (Stream and Gender)

Ensured column types are numeric for model compatibility

Maintained column order consistency for prediction phase

## Models Trained

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest Classifier ✅

XGBoost Classifier

Model Comparison (Accuracy):

Logistic Regression: 78.1%

KNN: 86.0%

SVM: 78.7%

Random Forest: 88.0%

XGBoost: 88.2%

## Final Model Selected

Random Forest Classifier

Saved using joblib

import joblib
joblib.dump(model, 'random_forest_model.pkl')

## Predictions on New Data

Created a function to safely predict placement outcome based on new input:

* Predict after aligning new data with trained feature order
prediction = model.predict(new_df[feature_order])

Example Output:

Predicted Placement Status: Placed

## Feature Importance

Top influential features:

CGPA

Internships

Stream (Computer Science, IT, etc.)

## Conclusion

Built a robust placement prediction pipeline

Compared multiple models for accuracy

Final model is ready for deployment or integration into a web app or dashboard

## Future Enhancements

Use cross-validation for more robust performance

Build a Streamlit web app for live prediction

✍️ Author

Vaibhav ShivhareB.Tech (CSE), Class of 2024Aspiring Data Scientist & ML Engineer
