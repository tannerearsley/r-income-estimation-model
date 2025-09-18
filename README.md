# r-income-estimation-model
- Adult Income Prediction | ML Classification with SVM, Logistic Regression, and Decision Trees 
- By Tanner Earsley  
- Created 9/14/2025

---

## Project Overview
- Use the UCI Adult Income dataset to predict whether an individual's income exceeds $50,000 annually.
- Perform data cleaning to handle missing values and remove unnecessary columns.
- Apply multiple supervised learning models (SVM, Logistic Regression, Decision Tree) to predict income.
- Compare model performance and determine which algorithm works best for this dataset.

---

## Methodology
- Data imported from .data file, with strings converted to factors and " ?" entries treated as missing values.
- Rows with missing values removed (~7% of data), and columns like FnlWgt and Education_Num removed because they are redundant or not impactful.
- Dataset split into training (80%) and test (20%) sets.
- Create confusion matrices for each model, then compare accuracies of each model in a dataframe.

---

## Included Files
- `income_prediction_ML_modeling.r` → R code for adult income modeling
- `adult.data` → initial adult income dataset
- `decision_tree_model.png` → Image of resultant decision tree model
- `model_accuracy_matrix.png` → Screenshot of accuracy matrix of each created model
- `user_estimate_predictions.png` → Estimate of user input values plugged into each model

---

## Notes
- Dataset comes from the 1994 U.S. Census Bureau (Current Population Survey) 
- Decision Tree visualization included for easier insight into predictions. 
- Completed entirely using R.

---
