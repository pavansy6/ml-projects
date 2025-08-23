import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

df = pd.read_csv(r"/home/pavan/Documents/PSY/ml-projects/decision_tree/data/customer_churn_dataset-testing-master.csv")

print(df.columns)
print(df.head())
print(df.info())
print(df.describe().T)

cat_cols = [col for col in df.columns if df[col].dtype == 'object']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
df.drop(['customerid'], axis=1, inplace=True)

x = df.drop(['churn'], axis=1)
y = df['churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Decision Tree
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(x_train, y_train)

print("Training Accuracy DT:", model.score(x_train, y_train))
print("Test Accuracy DT:", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# GridSearchCV
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 10],
    "criterion": ["gini", "entropy"]
}

gridsearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
gridsearch.fit(x_train, y_train)

print("Best Parameters:", gridsearch.best_params_)
print("Best CV Score:", gridsearch.best_score_)

best_dt = gridsearch.best_estimator_
y_pred = best_dt.predict(x_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Cross Validation Score
scores = cross_val_score(best_dt, x, y, cv=5, scoring="accuracy")
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())