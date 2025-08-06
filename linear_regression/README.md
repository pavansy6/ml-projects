# King County House Prices: Linear Models

This project applies **Linear Regression, Ridge Regression, and Lasso Regression** to predict house prices in King County.  

---

## 📊 Data Preprocessing

### 🔹 Log-Transforming the Target
House prices are highly skewed. To reduce heteroscedasticity, we use a log-transform:

![equation](https://latex.codecogs.com/svg.latex?y%20=%20\log(\text{price}))

---

### 🔹 Feature Engineering
We created new features to capture non-linear effects:

- **sqft_living × grade** → larger high-quality homes are disproportionately expensive.  
- **bath_bed_ratio = bathrooms / bedrooms** → reflects home livability.  

---

### 🔹 Standardization
For Ridge & Lasso, we standardize features:

![equation](https://latex.codecogs.com/svg.latex?z%20=%20\frac{x%20-%20\mu}{\sigma})

---

## 🧮 Mathematical Foundations

### 1. Linear Regression
We model house price as:

![equation](https://latex.codecogs.com/svg.latex?\hat{y}=\beta_0+\beta_1x_1+\beta_2x_2+\dots+\beta_nx_n)

Coefficients are estimated by minimizing the **Residual Sum of Squares (RSS):**

![equation](https://latex.codecogs.com/svg.latex?RSS=\sum_{i=1}^m(y_i-\hat{y}_i)^2)

---

### 2. Ridge Regression (L2 Regularization)
Adds a squared penalty term to shrink coefficients:

![equation](https://latex.codecogs.com/svg.latex?\text{Loss}=RSS+\alpha\sum_{j=1}^n\beta_j^2)

✅ Handles multicollinearity.  
🚫 Never sets coefficients to 0.  

---

### 3. Lasso Regression (L1 Regularization)
Adds an absolute penalty:

![equation](https://latex.codecogs.com/svg.latex?\text{Loss}=RSS+\alpha\sum_{j=1}^n|\beta_j|)

✅ Performs **feature selection** by setting some coefficients exactly to 0.  

---

## 📏 Evaluation Metrics

Since the model works in log-space, we evaluate in two ways:

### 🔹 Coefficient of Determination (R²)

![equation](https://latex.codecogs.com/svg.latex?R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2})

- Measures variance explained in log(price).  
- ![equation](https://latex.codecogs.com/svg.latex?R^2\approx0.77) → model explains ~77% of variance.  

---

### 🔹 Root Mean Squared Error (RMSE)
Evaluated in **original dollar prices**:

![equation](https://latex.codecogs.com/svg.latex?RMSE=\sqrt{\frac{1}{m}\sum(\exp(y_i)-\exp(\hat{y}_i))^2})

- RMSE ≈ **\$221,000** → predictions are off by ~$221K on average.  

---

## 📊 Results

| Model              | R² (log-price) | RMSE (original $) |
|--------------------|----------------|-------------------|
| Linear Regression  | 0.7666         | 221,362           |
| Ridge Regression   | 0.7666         | 221,543           |
| Lasso Regression   | 0.7664         | 223,247           |

✅ **Linear Regression** performs best.  
✅ **Ridge Regression** almost identical → confirms stability.  
✅ **Lasso Regression** slightly worse but useful for feature selection.  

---

## 🔍 Coefficient Insights
- Large **positive coefficients** (e.g., `sqft_living`, `grade`) → increase price.  
- Large **negative coefficients** (e.g., very high `bath_bed_ratio`) → decrease price.  
- Near-zero coefficients in Lasso → unimportant features.  

---

## 📉 Residual Analysis
Residuals show random scatter around 0 → no strong bias.  
Some heteroscedasticity remains (bigger errors for expensive homes), but log-transform improved performance significantly.  

---

## ✅ Summary
- Log-transform + interaction terms improved performance.  
- Regularization (Ridge/Lasso) confirmed model robustness.  
- Errors (~\$221K) are realistic given housing market variability.  
- Linear models provide a strong baseline before moving to tree-based methods.  

---
