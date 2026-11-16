
import pandas as pd
from pathlib import Path

# Load the dataset
file_path = Path(__file__).parent / "car_data.csv"
df = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset loaded successfully!")
print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
df.info()


from datetime import datetime

# Remove duplicate records
print("\nDuplicate records before cleaning:", df.duplicated().sum())

df = df.drop_duplicates().copy()

# Clean column names and car names
df.columns = df.columns.str.strip()
df["name"] = df["name"].astype(str).str.strip()

# Create a new feature: car age
current_year = datetime.now().year
df["car_age"] = current_year - df["year"]

# Display cleaned data
print("\nDataset shape after cleaning:", df.shape)

print("\nCar age column added successfully!")
print(df[["name", "year", "car_age", "selling_price"]].head())

# Save cleaned dataset
output_path = Path(__file__).parent / "cleaned_car_data.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved as cleaned_car_data.csv")


import matplotlib.pyplot as plt

# Graph 1: Distribution of car selling prices
plt.figure(figsize=(9, 5))
plt.hist(df["selling_price"], bins=30, edgecolor="black")
plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "price_distribution.png", dpi=300)
plt.show()

# Graph 2: Average selling price by fuel type
fuel_prices = df.groupby("fuel")["selling_price"].mean().sort_values(
    ascending=False
)

plt.figure(figsize=(9, 5))
fuel_prices.plot(kind="bar", edgecolor="black")
plt.title("Average Car Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "price_by_fuel.png", dpi=300)
plt.show()

print("\nAverage selling price by fuel type:")
print(fuel_prices.round(2))

print("\nEDA graphs saved successfully!")


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Select input features and target
features = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

X = df[features]
y = df["selling_price"]

# Separate categorical and numerical columns
categorical_features = [
    "name", "fuel", "seller_type", "transmission", "owner"
]

numerical_features = ["year", "km_driven"]

# Encode text columns and keep numerical columns unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("numerical", "passthrough", numerical_features)
    ]
)

# Create the machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
print("\nTraining the Random Forest model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Results")
print(f"Mean Absolute Error (MAE): Rs. {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# Save the trained model
model_path = Path(__file__).parent / "car_price_model.pkl"
joblib.dump(model, model_path)

print("\nTrained model saved as car_price_model.pkl")


# Test the model with a sample car

sample_car = pd.DataFrame([{
    "name": "Maruti Swift Dzire VDI",
    "year": 2017,
    "km_driven": 60000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner"
}])

predicted_price = model.predict(sample_car)[0]

print("\n--- Sample Car Price Prediction ---")
print("Car: Maruti Swift Dzire VDI")
print("Year: 2017")
print("Kilometres Driven: 60,000")
print(f"Predicted Selling Price: Rs. {predicted_price:,.2f}")
