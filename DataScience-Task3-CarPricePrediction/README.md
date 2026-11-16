
# Car Price Prediction with Machine Learning

## Project Overview
This project uses machine learning to predict the selling price of a used car based on its features, including manufacturing year, kilometres driven, fuel type, seller type, transmission, and ownership history.

## Objectives
- Explore and clean used-car data.
- Analyze selling prices using visualizations.
- Train a Random Forest Regression model.
- Evaluate model performance.
- Predict the selling price of a sample car.

## Dataset
**File:** `car_data.csv`

The dataset contains 4,340 records initially and 8 columns, including:
- Car name
- Year
- Selling price
- Kilometres driven
- Fuel
- Seller type
- Transmission
- Owner

Duplicate rows are removed during preprocessing, so the cleaned record count may be lower.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Visualizations
- `price_distribution.png` — Distribution of selling prices.
- `price_by_fuel.png` — Average selling price by fuel type.

## Machine Learning Model
**Algorithm:** Random Forest Regressor

The data is divided into training and testing sets using an 80:20 split.

### Model Evaluation
- R² Score: 0.5155
- Mean Absolute Error (MAE): ₹1,50,088.47
- Training records: 2,861
- Testing records: 716

### Sample Prediction
For a 2017 Maruti Swift Dzire VDI with 60,000 km driven, the model predicted a selling price of **₹6,30,800**.

This is a model estimate, not a guaranteed market price.

## How to Run
Install the required packages:

```bash
pip install pandas matplotlib scikit-learn joblib
```

Keep `car_data.csv` in the same folder as `car_price_prediction.py`, then run:

```bash
python car_price_prediction.py
```

The script generates visualizations, evaluates the model, and saves the trained model as `car_price_model.pkl`.

## Conclusion
This project demonstrates how machine learning can estimate used-car prices from vehicle attributes. Model performance can be improved through feature engineering and hyperparameter tuning.

## Author
Shreya Kumari
