# Customer Churn Prediction: Model & Metrics

This document provides an overview of the customer churn prediction model, including data preprocessing with StandardScaler, the mathematical foundations of Logistic Regression, and the key metrics used to evaluate its performance.

## Data Preprocessing: StandardScaler

StandardScaler is a crucial preprocessing step that standardizes numerical features by rescaling them to have a **mean of 0** and a **standard deviation of 1**.

### What Does It Do?

For each feature, it transforms every value ($x$) using the following formula, also known as a Z-score:

$$z = \frac{x - \mu}{\sigma}$$

Where:
* $\mu$ is the mean of the feature column.
* $\sigma$ is the standard deviation of the feature column.

### Why Is It Important?

Machine learning algorithms like Logistic Regression can be sensitive to the scale of input features. For example, a feature like `balance` (ranging from 0 to 250,000) would have a much larger impact on the model than a feature like `numofproducts` (ranging from 1 to 4) simply because of its larger scale.

Standardization ensures that all features contribute more equally to the model's training, which often leads to better model performance and faster convergence.

---

## The Mathematics Behind Logistic Regression

Logistic Regression is a classification algorithm used to predict a binary outcome (like churn vs. no-churn). While it shares a name with Linear Regression, it's fundamentally different because it predicts probabilities.

The core idea is to transform a linear equation into a probability using a **Sigmoid function**.

### 1. The Linear Equation (Logit)

First, the model calculates a linear combination of the input features ($x_1, x_2, ...$) and their corresponding weights ($\beta_1, \beta_2, ...$), plus a bias term ($\beta_0$). This output, often called the **logit**, can be any real number.

$$z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$$

### 2. The Sigmoid Function

This linear output $z$ is then passed through the Sigmoid (or Logistic) function, $\sigma(z)$. This function squashes any real-valued number into a range between 0 and 1, making it a valid probability.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

* If $z$ is a large positive number, $e^{-z}$ approaches 0, and the output $\sigma(z)$ approaches 1.
* If $z$ is a large negative number, $e^{-z}$ becomes very large, and the output $\sigma(z)$ approaches 0.
* If $z$ is 0, the output $\sigma(z)$ is exactly 0.5.



[Image of a sigmoid function curve]


### 3. Making a Prediction

The output of the sigmoid function is the predicted probability, $\hat{p}$. To make a final classification, we use a threshold (typically 0.5):
* If $\hat{p} \ge 0.5$, predict class 1 (Churn).
* If $\hat{p} < 0.5$, predict class 0 (No Churn).

The model "learns" by finding the optimal values for the coefficients ($\beta_0, \beta_1, ...$) that minimize a cost function (like Log Loss) using an optimization algorithm like Gradient Descent.

---

## Model Evaluation Metrics

To understand how well our model performs, we use several key metrics derived from the **Confusion Matrix**.

### The Confusion Matrix

The Confusion Matrix is a table that summarizes the performance of a classification model by comparing the actual and predicted values.



* **True Positives (TP)**: The model correctly predicted a customer would **churn**.
* **True Negatives (TN)**: The model correctly predicted a customer would **not churn**.
* **False Positives (FP)**: The model incorrectly predicted a customer would **churn** (a "false alarm").
* **False Negatives (FN)**: The model incorrectly predicted a customer would **not churn** (it "missed" a churner).

All other metrics are calculated from these four values.

### 1. Accuracy

Accuracy measures the overall correctness of the model.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Use Case**: A general measure of performance.
**Warning**: It can be misleading for imbalanced datasets. For example, if 95% of customers don't churn, a model that always predicts "no churn" is 95% accurate but useless.

### 2. Precision

Precision answers the question: "When the model predicts a customer will churn, how often is it correct?"

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Use Case**: Precision is important when the cost of a **False Positive** is high. For example, if you spend a lot of money on retention offers, you want to be sure the customers you target are actually at risk of churning.

### 3. Recall (Sensitivity)

Recall answers the question: "Of all the customers who actually churned, how many did the model find?"

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Use Case**: Recall is crucial when the cost of a **False Negative** is high. In churn prediction, failing to identify a customer who is about to leave (a False Negative) is a significant lost opportunity. **This is often the most important metric for a churn project.**

### 4. F1-Score

The F1-Score is the harmonic mean of Precision and Recall, providing a single score that balances both.

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Use Case**: It's useful when you need a balance between Precision and Recall and don't want to focus on one metric alone. It punishes models that have a very high score for one metric at the expense of the other.