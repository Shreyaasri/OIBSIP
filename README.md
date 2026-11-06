# Iris Flower Classification using Machine Learning

## Project Overview
This project classifies Iris flowers into three species: Setosa, Versicolor, and Virginica using machine learning algorithms in Python.

## Objectives
- Explore and analyze the Iris dataset.
- Check missing values and understand data distributions.
- Visualize flower measurements.
- Train and evaluate machine learning models.
- Compare Logistic Regression and K-Nearest Neighbors (KNN).

## Dataset
The project uses the built-in Iris dataset from Scikit-learn.

- Total samples: 150
- Input features: 4
- Target classes: 3
- Training samples: 120
- Testing samples: 30

### Input Features
- Sepal length
- Sepal width
- Petal length
- Petal width

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Machine Learning Models
1. Logistic Regression
2. K-Nearest Neighbors (KNN)

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.67% |
| KNN | 100.00% |

These results are from the current 80/20 train-test split with `random_state=42`. Accuracy may vary with a different split or dataset.

## Visualizations
- `confusion_matrix.png` — Confusion matrix for Logistic Regression.
- `model_comparison.png` — Accuracy comparison between both models.

## How to Run

1. Install Python.
2. Install the required libraries:

   `python -m pip install scikit-learn pandas matplotlib seaborn`

3. Run the project from the parent `OIBSIP` folder:

   `python DataScience-Task1-IrisClassification/iris_classification.py`

## Author
Shreya Kumari

## Internship
Oasis Infobyte — Data Science Internship
