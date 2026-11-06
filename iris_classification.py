# Iris Flower Classification Project

from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert dataset into a DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add flower species as a column
df["species"] = iris.target_names[iris.target]

# Display the first five rows
print("First 5 rows of the Iris Dataset:")
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nSpecies Names:")
print(iris.target_names)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Count flowers in each species
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="species")

plt.title("Number of Flowers in Each Species")
plt.xlabel("Flower Species")
plt.ylabel("Number of Flowers")

plt.tight_layout()
#plt.show()

# Compare petal measurements by species
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    s=70
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend(title="Species")
plt.tight_layout()
#plt.show()

# Step 10: Split the dataset

from sklearn.model_selection import train_test_split

# Input features and target labels
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training labels:", len(y_train))
print("Testing labels:", len(y_test))

# Step 11: Train the Logistic Regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Model Results:")
print("Correct predictions:", sum(y_pred == y_test), "out of", len(y_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Step 12: Model Evaluation

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix - Iris Classification")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()

plt.savefig("DataScience-Task1-IrisClassification/confusion_matrix.png")

# Step 13: Train a KNN model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict flower species
knn_pred = knn_model.predict(X_test)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Model Results:")
print("Correct predictions:", sum(knn_pred == y_test), "out of", len(y_test))
print("KNN Accuracy:", round(knn_accuracy * 100, 2), "%")

print("\nModel Comparison:")
print("Logistic Regression:", round(accuracy * 100, 2), "%")
print("KNN:", round(knn_accuracy * 100, 2), "%")

# Step 14: Compare model accuracy visually

import matplotlib.pyplot as plt

models = ["Logistic Regression", "KNN"]
accuracies = [accuracy * 100, knn_accuracy * 100]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)

for bar, score in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{score:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "DataScience-Task1-IrisClassification/model_comparison.png",
    dpi=300
)

print("\nModel comparison graph saved successfully!")