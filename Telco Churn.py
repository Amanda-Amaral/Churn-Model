# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# %%
df = pd.read_csv('Telco_Customer_Churn.csv')

# %% [markdown]
# #### General Information

# %%
df.info()

# %% [markdown]
# #### Columns Header

# %%
df.head()

# %% [markdown]
# #### Check for Missing Values

# %%
df.isnull().sum()

# %% [markdown]
# #### A Partial Correlation Heatmap

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# #### Converting Categorical Values to Numerical

# %%
df = pd.get_dummies(df, drop_first=True)
df.head()

# %% [markdown]
# #### Note that some rows (every row which has anything else besides numbers) became columns. 
# Such as "customerID_0003-MKNFE" or "Partner" per example
# #### Notice that the column "TotalCharges" has only numbers but it was converted too, wrongly. That might happen because the number has a point to separate the decimals or because the numbers are stored as string or object. Let's check and deal with it

# %%
df = pd.read_csv('Telco_Customer_Churn.csv')
df["TotalCharges"]

# %% [markdown]
# #### Converting to numerical (float) using "to_numeric"

# %%
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"]

# %% [markdown]
# #### Convert Categorical Values to Numerical
# ##### Notice that the extra columns related to the "ToTalCharges" just disappeared, standing only the original
# From 13602 columns it reduces to 7073 columns

# %%
df = pd.get_dummies(df, drop_first=True)
df.head()

# %% [markdown]
# #### Data Visualization

# %%
sns.countplot(x='Churn_Yes', data=df)
plt.title('Churn Count')
plt.show()

# %% [markdown]
# #####  Number 1 represents the customer who left within last month

# %%
sns.boxplot(x='Churn_Yes', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.show()

# %% [markdown]
# ##### Higher the Monthly Charges, there are more people unsubscribing and the opposite happen too. The lower the Monthly Charges, there are more people continuing the subscription

# %%
fig = px.histogram(df, x='tenure', color='Churn_Yes', title='Customer Tenure Distribution by Churn Status')
fig.show()

# %% [markdown]
# ##### We can observe that as tenure decreases, the churn rate increases

# %%
fig = px.box(df, x='Churn_Yes', y='MonthlyCharges', title='Distribution of Monthly Charges by Churn Status')
fig.show()

# %% [markdown]
# ##### On this graphic we can observe that the range of Monthly Charges from Churn or not overlap partially and the median from both are not that far

# %% [markdown]
# #### Checking once again for missing values

# %%
print(df.isnull().sum())

# %% [markdown]
# #### Dealing with missing values

# %%
df.fillna(df.mean(), inplace=True)

# %% [markdown]
# #### Confirming the elimination of missing values

# %%
print(df.isnull().sum())

# %% [markdown]
# ##### Checking the DType of each Column

# %%
print(df.dtypes)

# %% [markdown]
# ##### Solving the problem of dtype unit8 which is not a number and can't be used in the model

# %%
def convert_uint8(column):
    column = np.uint8(column)
    new_column = np.int8(column)
    return new_column

# %%
for column in df:
    if (df[column].dtypes) == "uint8":
        df[column] = convert_uint8(df[column])

# %%
print(df.dtypes)

# %% [markdown]
# #### Handling any missing value

# %%
df.dropna(inplace=True)

# %% [markdown]
# #### THE MODEL
# ##### Time to split the data into Training and Testing Sets

# %%
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# #### Logistic Regression Model

# %%
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# %% [markdown]
# #### Making predictions

# %%
y_pred = model.predict(X_test)

# %% [markdown]
# #### Evaluating the accuracy of the model

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# #### - Zero id related to the negative class (No Churn)
# #### - One is related to the positive class

# %% [markdown]
# #### Making the Confusion Matrix Friendly

# %%
sns.heatmap(confusion_matrix(y_test, y_pred)/np.sum(confusion_matrix(y_test, y_pred)), annot=True, fmt='.2%', cmap='Blues')

# %% [markdown]
# #### - The top left square represents the True Negatives which means the % of correct in guessing a "No Churn"
# #### - The top right square represents the False Positives which means the % of incorrect in guessing "Churn"
# #### - The down left square represents the False Negatives which means the % of incorrect in guessing "No Churn"
# #### - The down right square represents the True Positives which means the % of correct in guessing a "Churn"

# %% [markdown]
# #### MODEL VALIDATION

# %% [markdown]
# #### This kind of validation splits the data in train/test sets (StratifiedKFold splitting in 5 data sets).
# #### To avoid overfitting, the train data flows from fold 1 to 5 along the data split, this way the algorithm is dubjected to 5 different datas to train and test.
# ![image.png](attachment:3ad59467-e7e4-4677-85ec-ac7979a9374d.png)
# #### source: https://scikit-learn.org/stable/modules/cross_validation.html

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print("Cross-Validation Accuracy Scores (Logistic Model):", cross_val_scores)
print("Mean Cross-Validation Accuracy (Logistic Model):", cross_val_scores.mean())

# %%
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cross_val_scores) + 1), cross_val_scores, marker='o')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy Scores (Logistic Model)')
plt.show()

# %% [markdown]
# #### The result looks nice but let's try another method and compare both

# %%
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

# %%
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Convert regression predictions to binary classification
y_pred_lasso_class = [1 if prob > 0.5 else 0 for prob in y_pred_lasso]

# %%
print("Lasso Regression:")
print(confusion_matrix(y_test, y_pred_lasso_class))
print(classification_report(y_test, y_pred_lasso_class))

# %%
sns.heatmap(confusion_matrix(y_test, y_pred_lasso_class)/np.sum(confusion_matrix(y_test, y_pred_lasso_class)), annot=True, fmt='.2%', cmap='Blues')

# %% [markdown]
# ### Observing the result numbers logistic model fits better for this dataset

# %%



