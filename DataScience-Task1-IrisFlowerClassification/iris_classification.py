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


# Step 11: Prepare the data for model training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path

# Save output files in the current project folder
project_dir = Path(__file__).resolve().parent

# Input features and target labels
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 12: Train both models

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

accuracies = []
results = {}

for name, model in models.items():

    # Use scaled features for Logistic Regression
    # and original features for Decision Tree
    if name == "Logistic Regression":
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test

    # Train the model
    model.fit(X_train_model, y_train)

    # Make predictions
    predictions = model.predict(X_test_model)
    results[name] = predictions

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy * 100)

    print(f"\n{name} Results:")
    print(
        "Correct predictions:",
        (predictions == y_test).sum(),
        "out of",
        len(y_test)
    )
    print("Accuracy:", round(accuracy * 100, 2), "%")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.tight_layout()

    filename = (
        "confusion_matrix.png"
        if name == "Logistic Regression"
        else "confusion_matrix_decision_tree.png"
    )
    plt.savefig(project_dir / filename, dpi=300)
    plt.close()

# Step 13: Compare model accuracy

print("\nModel Comparison:")
for name, score in zip(models.keys(), accuracies):
    print(f"{name}: {score:.2f}%")

plt.figure(figsize=(8, 5))
bars = plt.bar(models.keys(), accuracies)

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
plt.savefig(project_dir / "model_comparison.png", dpi=300)
plt.close()

print("\nAll evaluation graphs saved successfully!")
