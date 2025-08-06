import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"/home/pavan/Documents/PSY/ml-projects/linear_regression/data/kc_house_data.csv")

print(df.columns)
print(df.head())
print(df.info())
print(df.describe().T)

# Drop irrelevant features
df.drop(columns=['id','zipcode','long','condition'], inplace=True, errors="ignore")

# Log-transform target
df["log_price"] = np.log(df["price"])

# Select Base Features
features = [
    "sqft_living", "grade", "sqft_above", "sqft_living15",
    "bathrooms", "view", "sqft_basement", "bedrooms",
    "lat", "waterfront", "floors", "yr_renovated",
    "sqft_lot", "sqft_lot15", "yr_built"
]

# Add engineered features
df["sqft_living_grade"] = df["sqft_living"] * df["grade"]
df["bath_bed_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1e-5)

features += ["sqft_living_grade", "bath_bed_ratio"]

x = df[features]
y = df["log_price"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Helper function
def evaluate_model(name, model, x_train, x_test, y_train, y_test, results):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))
    results.append([name, r2, rmse])
    return model


# Baseline Linear Regression
results = []
lr = LinearRegression()
evaluate_model("Linear Regression", lr, x_train, x_test, y_train, y_test, results)


# Ridge Regression with CV
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
ridge_params = {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(ridge_pipe, ridge_params, cv=5, scoring="r2")
ridge_model = evaluate_model("Ridge Regression", ridge_cv, x_train, x_test, y_train, y_test, results)


# Lasso Regression with CV
lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=10000))
])
lasso_params = {"lasso__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
lasso_cv = GridSearchCV(lasso_pipe, lasso_params, cv=5, scoring="r2")
lasso_model = evaluate_model("Lasso Regression", lasso_cv, x_train, x_test, y_train, y_test, results)


# Results Table
results_df = pd.DataFrame(results, columns=["Model", "R2 (log-price)", "RMSE (original $)"])
print(results_df)


# Coefficient Inspection (Lasso)
best_lasso = lasso_model.best_estimator_.named_steps["lasso"]
coef = pd.Series(best_lasso.coef_, index=x.columns)
print("\nLasso Coefficients:")
print(coef.sort_values())


# Residual Plot
y_pred_lr = lr.predict(x_test)
residuals = y_test - y_pred_lr

plt.scatter(y_pred_lr, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted log(price)")
plt.ylabel("Residuals (log space)")
plt.title("Residual Plot - Linear Regression")
plt.show()