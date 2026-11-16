
import pandas as pd
from pathlib import Path

# Locate the CSV file in this project folder
data_path = Path(__file__).parent / "Unemployment in India.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove completely empty rows
df = df.dropna(how="all")


# Convert Date column to datetime format
df["Date"] = pd.to_datetime(
    df["Date"].astype(str).str.strip(),
    dayfirst=True,
    errors="coerce"
)

# Convert unemployment-related columns to numeric
numeric_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nData types after cleaning:")
print(df.dtypes)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Display basic information
print("UNEMPLOYMENT DATASET ANALYSIS")
print("-" * 35)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
df.info()


# Step 9: Visualize unemployment rate over time
import matplotlib.pyplot as plt

# Calculate average unemployment rate for each date
trend_data = (
    df.dropna(subset=["Date", "Estimated Unemployment Rate (%)"])
    .groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_index()
)

# Create the trend graph
plt.figure(figsize=(12, 6))

plt.plot(
    trend_data.index,
    trend_data.values,
    marker="o",
    linewidth=2
)

plt.title("Unemployment Rate in India Over Time")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the graph in the Task 2 folder
graph_path = Path(__file__).parent / "unemployment_trend.png"
plt.savefig(graph_path, dpi=300)

print("\nUnemployment trend graph saved at:", graph_path)

plt.show()


# Step 11: Compare average unemployment rates by region

region_data = (
    df.dropna(subset=["Region", "Estimated Unemployment Rate (%)"])
    .groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(12, 8))

region_data.plot(kind="barh")

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.tight_layout()

# Save the graph in the Task 2 folder
region_graph_path = Path(__file__).parent / "region_unemployment.png"
plt.savefig(region_graph_path, dpi=300)

print("\nRegional unemployment graph saved at:", region_graph_path)

plt.show()

print("\nHighest average unemployment rate:")
print(region_data.idxmax(), "-", round(region_data.max(), 2), "%")

print("\nLowest average unemployment rate:")
print(region_data.idxmin(), "-", round(region_data.min(), 2), "%")


# Step 12: Compare unemployment before and during COVID-19

# Keep rows with valid dates and unemployment rates
covid_df = df.dropna(
    subset=["Date", "Estimated Unemployment Rate (%)"]
).copy()

# Divide data into pre-COVID and COVID periods
covid_df["Period"] = covid_df["Date"].apply(
    lambda date: "Before COVID-19"
    if date < pd.Timestamp("2020-03-01")
    else "During COVID-19"
)

# Calculate average unemployment rate for each period
covid_comparison = covid_df.groupby("Period")[
    "Estimated Unemployment Rate (%)"
].mean()

# Display results
print("\nCOVID-19 Impact Analysis")
print("-" * 30)
print(covid_comparison.round(2))

# Create comparison graph
plt.figure(figsize=(8, 5))
covid_comparison.plot(kind="bar")

plt.title("Unemployment Rate: Before vs During COVID-19")
plt.xlabel("Period")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

# Save graph
covid_graph_path = Path(__file__).parent / "covid_impact.png"
plt.savefig(covid_graph_path, dpi=300)

print("\nCOVID-19 comparison graph saved at:", covid_graph_path)

plt.show()



# Step 13: Compare unemployment in Urban and Rural areas

area_data = (
    df.dropna(subset=["Area", "Estimated Unemployment Rate (%)"])
    .groupby("Area")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment Rate by Area:")
print(area_data.round(2))

plt.figure(figsize=(8, 5))
area_data.plot(kind="bar")

plt.title("Average Unemployment Rate: Urban vs Rural")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

# Save the graph
area_graph_path = Path(__file__).parent / "area_unemployment.png"
plt.savefig(area_graph_path, dpi=300)

print("\nArea comparison graph saved at:", area_graph_path)

plt.show()


# Step 14: Final Project Observations

print("\n" + "=" * 45)
print("FINAL PROJECT OBSERVATIONS")
print("=" * 45)

print(
    f"\nHighest average unemployment region: "
    f"{region_data.idxmax()} ({region_data.max():.2f}%)"
)

print(
    f"Lowest average unemployment region: "
    f"{region_data.idxmin()} ({region_data.min():.2f}%)"
)

print("\nAverage Unemployment Before and During COVID-19:")
print(covid_comparison.round(2))

if (
    "Before COVID-19" in covid_comparison.index
    and "During COVID-19" in covid_comparison.index
):
    before = covid_comparison["Before COVID-19"]
    during = covid_comparison["During COVID-19"]

    print(f"\nChange in average unemployment rate: {during - before:.2f} percentage points")

print("\nAverage Unemployment by Area:")
print(area_data.round(2))

print("\nAll visualizations have been generated.")


# Unemployment trends across the top 3 regions

top_3_regions = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .nlargest(3)
    .index
)

region_trend = df[df["Region"].isin(top_3_regions)]

plt.figure(figsize=(12, 6))

for region in top_3_regions:
    region_data = region_trend[region_trend["Region"] == region]
    monthly_data = (
        region_data.groupby("Date")["Estimated Unemployment Rate (%)"]
        .mean()
        .sort_index()
    )

    plt.plot(
        monthly_data.index,
        monthly_data.values,
        marker="o",
        label=region
    )

plt.title("Unemployment Trends Across Top 3 Regions")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("top_3_region_trends.png", dpi=300)
plt.show()

print("\nTop 3 regions trend graph saved successfully!")


# Correlation Heatmap

import matplotlib.pyplot as plt

correlation_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

correlation_data = df[correlation_columns].corr()

plt.figure(figsize=(9, 6))

image = plt.imshow(correlation_data, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(image, label="Correlation")

plt.xticks(
    range(len(correlation_columns)),
    ["Unemployment Rate", "Estimated Employed", "Labour Participation"],
    rotation=25,
    ha="right"
)

plt.yticks(
    range(len(correlation_columns)),
    ["Unemployment Rate", "Estimated Employed", "Labour Participation"]
)

for i in range(len(correlation_columns)):
    for j in range(len(correlation_columns)):
        plt.text(
            j, i,
            f"{correlation_data.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )

plt.title("Correlation Heatmap of Unemployment Indicators")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

print("\nCorrelation heatmap saved successfully!")
print(correlation_data.round(2))
