
# Unemployment Analysis with Python

## Project Overview
This project analyzes unemployment trends in India using Python and data visualization. It explores unemployment rates across regions, compares unemployment before and during COVID-19, and examines differences between urban and rural areas.

## Objectives
- Analyze unemployment trends over time.
- Identify regions with the highest and lowest average unemployment rates.
- Compare unemployment before and during COVID-19.
- Compare urban and rural unemployment rates.
- Visualize relationships between unemployment, employment, and labour participation.

## Dataset
**File:** `Unemployment in India.csv`

The dataset includes:
- Region
- Date
- Frequency
- Estimated Unemployment Rate (%)
- Estimated Employed
- Estimated Labour Participation Rate (%)
- Area

## Technologies Used
- Python
- Pandas
- Matplotlib
- VS Code

## Analysis and Visualizations
The project generates the following visualizations:

1. `unemployment_trend.png` — Unemployment trend over time.
2. `region_unemployment.png` — Average unemployment by region.
3. `covid_impact.png` — Before and during COVID-19 comparison.
4. `area_unemployment.png` — Urban versus rural unemployment.
5. `top_3_region_trends.png` — Trends for the three regions with the highest average unemployment.
6. `correlation_heatmap.png` — Correlation between unemployment, employment, and labour participation indicators.

## Key Findings
Based on the current analysis:

- Highest average unemployment region: Tripura (28.35%).
- Lowest average unemployment region: Meghalaya (4.80%).
- Average unemployment before COVID-19: 9.51%.
- Average unemployment during COVID-19: 17.77%.
- Increase between the two periods: 8.26 percentage points.
- Average urban unemployment: 13.17%.
- Average rural unemployment: 10.32%.

These results are based on the dataset and the analysis performed in this project.

## How to Run
1. Install Python.
2. Install the required libraries:

   ```bash
   pip install pandas matplotlib
   ```

3. Keep `Unemployment in India.csv` and `unemployment_analysis.py` in the same folder.
4. Run the script:

   ```bash
   python unemployment_analysis.py
   ```

## Conclusion
This project demonstrates how Python can be used to analyze unemployment data, identify regional differences, compare time periods, and communicate findings through visualizations.

## Author
Shreya Kumari
