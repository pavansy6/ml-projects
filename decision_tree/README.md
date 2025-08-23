# 📊 Customer Churn Prediction: Decision Tree Classifier

This project applies a **Decision Tree Classifier** with **GridSearchCV** and **Cross-Validation** to predict customer churn.  

---

## 🔹 Data Preprocessing

1. **Encoding Categorical Variables**  
   All categorical columns were label encoded using `LabelEncoder`.  
   Example:  
   Yes -> 1
   No -> 0

2. **Feature Engineering**  
- Removed unnecessary identifiers (`customerID`).  
- Standardized column names for consistency.  

3. **Train-Test Split**  
Dataset was split:  
- **80%** for training  
- **20%** for testing  

---

## 🧮 Mathematical Foundations

### 1. Decision Tree Classifier
A decision tree splits data based on feature thresholds to minimize **impurity**.  

At each split, the algorithm chooses the feature/threshold that maximizes **information gain**:

![equation](https://latex.codecogs.com/svg.latex?IG(D,f)=H(D)-\sum_{v\in Values(f)}\frac{|D_v|}{|D|}H(D_v))

Where:  
- ![equation](https://latex.codecogs.com/svg.latex?H(D)) = entropy of dataset  
- ![equation](https://latex.codecogs.com/svg.latex?D_v) = subset after splitting on feature `f`  

Entropy is defined as:  

![equation](https://latex.codecogs.com/svg.latex?H(D)=-\sum_{i=1}^k p_i \log_2(p_i))

---

### 2. GridSearchCV
GridSearch systematically tests hyperparameter combinations to find the best performing model.  

We tuned:  
- `max_depth` (depth of tree)  
- `min_samples_split` (minimum samples to split a node)  
- `min_samples_leaf` (minimum samples per leaf)  
- `criterion` (`gini` or `entropy`)  

---

### 3. Cross-Validation
To avoid overfitting, we used **k-fold cross-validation** (k=5):  

![equation](https://latex.codecogs.com/svg.latex?CV=\frac{1}{k}\sum_{i=1}^k\text{Accuracy}_i)

---

## 📏 Evaluation Metrics

### 🔹 Accuracy
![equation](https://latex.codecogs.com/svg.latex?Accuracy=\frac{TP+TN}{TP+FP+TN+FN})

### 🔹 Precision
![equation](https://latex.codecogs.com/svg.latex?Precision=\frac{TP}{TP+FP})

### 🔹 Recall
![equation](https://latex.codecogs.com/svg.latex?Recall=\frac{TP}{TP+FN})

### 🔹 F1-Score
![equation](https://latex.codecogs.com/svg.latex?F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall})

---

## 📊 Results

| Metric              | Value |
|----------------------|-------|
| Training Accuracy    | ~0.998 |
| Test Accuracy        | ~0.999 |
| Best Parameters      | {'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 5} |
| Mean CV Accuracy     | ~0.9989 |

---

## ✅ Summary

- **Decision Tree** achieved extremely high accuracy on both train and test sets.  
- **GridSearchCV** optimized hyperparameters for best performance.  
- **Cross-validation** confirmed model stability (very low variance).  
- Provides a strong baseline before moving to more complex ensemble methods (Random Forest, XGBoost).  

---
