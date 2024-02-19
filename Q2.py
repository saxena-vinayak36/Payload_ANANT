# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target
target_names = iris_data.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
logistic_reg_model = LogisticRegression(max_iter=1000)
logistic_reg_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_reg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Visualize Actual vs Predicted Classes
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolors='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Actual vs Predicted Classes')
plt.show()

# Cross-validated predictions
cv_predictions = cross_val_predict(logistic_reg_model, X, y, cv=5)

# Evaluate cross-validated predictions
cv_accuracy = accuracy_score(y, cv_predictions)
cv_classification_rep = classification_report(y, cv_predictions, target_names=target_names)

print("Cross-validated Accuracy Score:", cv_accuracy)
print("\nCross-validated Classification Report:\n", cv_classification_rep)

# Visualize Cross-validated Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y, cv_predictions), annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Cross-validated Confusion Matrix')
plt.show()