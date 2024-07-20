# Telco Customer Churn Prediction
This project aims to predict customer churn in a telecommunications company using Logistic Regression and Lasso Regression models. The Telco Customer Churn dataset from Kaggle is used for this purpose. The project includes data preprocessing, model training, evaluation, and comparison of both models.

## Dataset
The dataset used in this project is the Telco Customer Churn dataset, which can be downloaded from Kaggle.

## Data Preprocessing
1.	Handle Missing Values: Remove rows with missing values.
2.	Convert Categorical Variables: Convert categorical variables to numerical using pd.get_dummies.
   
## Model Training
### Logistic Regression
Trains a logistic regression model to classify customer churn.
### Lasso Regression
Trains a Lasso regression model to predict the probability of churn, then converts the probabilities to binary classification.
## Model Evaluation
Both models are evaluated using confusion matrices, classification reports, and accuracy scores. The accuracy of both models is compared to determine the better-performing model.
## Conclusion
This project demonstrates how to preprocess data, train, and evaluate logistic and Lasso regression models for predicting customer churn. Logistic Regression typically performs better for classification tasks, while Lasso Regression helps with feature selection by penalizing less significant features.
 
