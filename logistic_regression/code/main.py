import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = '/home/pavan/Documents/PSY/customer_churn/data/dataset.mat'

# If the file path is relative, adjust it accordingly
with open(file_path, 'r') as f:
    data_dict = arff.load(f)

# This restores 'df' to a Pandas DataFrame
df = pd.DataFrame(data_dict['data'])
df.columns = [attr[0] for attr in data_dict['attributes']]

print("DataFrame reloaded successfully!")
# print(df.head())

# Preprocess the data

# 1. Separate your features (X) and target (y)
y = df['exited']
x = df.drop('exited', axis=1)

# 2. Select numerical columns from X
numerical_cols = x.select_dtypes(include=['float64', 'int64']).columns

# 3. Initialize and apply the scaler
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x[numerical_cols]), columns=numerical_cols)

# Check the result
print("\nScaling complete. Scaled features head:")
# print(X_scaled.head())

# 4. Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# 5. Initialize and train the logistic regression model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(x_train, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(x_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.2f}")

# 8. Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))