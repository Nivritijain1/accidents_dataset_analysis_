import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch


# Load dataset
df = pd.read_csv("accident.csv", encoding='latin1')

# Drop column with excessive missing values
if 'TWAY_ID2' in df.columns:
    df.drop(columns=['TWAY_ID2'], inplace=True)

# Create derived metrics
df['VEH_PER_PERSON'] = df['VE_TOTAL'] / (df['PERSONS'] + 0.1)
df['TIME_DELAY'] = (df['ARR_HOUR'] - df['HOUR']).replace(99, np.nan)

# Keep only valid HOUR entries (0-23)
df = df[df['HOUR'].between(0, 23)]

# Check dataset size
print("Total records after cleaning:", df.shape[0])

# Check for missing values
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# Check for duplicate records
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Descriptive statistics for key numerical columns
print("\nDescriptive Statistics:")
print(df[['FATALS', 'VE_TOTAL', 'PERSONS', 'VEH_PER_PERSON', 'TIME_DELAY']].describe())

# Correlation matrix using Pearson's correlation
print("\nCorrelation Matrix:")
print(df[['FATALS', 'VE_TOTAL', 'PERSONS', 'VEH_PER_PERSON', 'TIME_DELAY']].corr(method='pearson'))



# 3. OBJECTIVE 1: Temporal Fatality Patterns (Hour of Day)

# hour data must be between 0 to 24 hours

df_cleaned = df[df["HOUR"].between(0, 23)]
'''
plt.figure(figsize=(10, 5)) 
sns.lineplot(
    x="HOUR",
    y="FATALS",
    data=df_cleaned.groupby("HOUR")["FATALS"].sum().reset_index()
)
plt.title("Total Fatalities by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Total Fatalities")
plt.tight_layout()
plt.show()
'''

# OBJECTIVE 2: Interaction of Weather and Light Conditions
'''

# --- Cleaning relevant columns ---
# Retaining rows with valid weather and light conditions
df_weather_light = df_cleaned[
    df_cleaned["WEATHERNAME"].notnull() & 
    df_cleaned["LGT_CONDNAME"].notnull()
]

# --- Create pivot table ---
pivot_table = df_cleaned.pivot_table(
    index="WEATHERNAME",
    columns="LGT_CONDNAME",
    values="FATALS",
    aggfunc="mean"  
)

# --- heatmap ---
plt.figure(figsize=(12, 7))
sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Average Fatalities'}
)

plt.title("🔍 Average Fatalities by Weather and Lighting Conditions", fontsize=14, weight='bold')
plt.xlabel("Lighting Conditions")
plt.ylabel("Weather Conditions")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
'''

#OBJECTIVE 3: Spatial Crash Clustering (Lat/Lon + Rural/Urban)

# Filter data
df = df[
    (df["LATITUDE"].between(-90, 90)) &
    (df["LONGITUD"].between(-180, 180)) &
    (df["RUR_URBNAME"].isin(["Urban", "Rural"]))
]

# Set plot size
plt.figure(figsize=(10, 6))

# Plot KDE for Urban
sns.kdeplot(
    data=df[df["RUR_URBNAME"] == "Urban"],
    x="LONGITUD", y="LATITUDE",
    fill=True, cmap="Blues", alpha=0.5
)

# Plot KDE for Rural
sns.kdeplot(
    data=df[df["RUR_URBNAME"] == "Rural"],
    x="LONGITUD", y="LATITUDE",
    fill=True, cmap="Greens", alpha=0.5
)

# Add custom legend
legend_patches = [
    Patch(facecolor='blue', edgecolor='blue', label='Urban'),
    Patch(facecolor='green', edgecolor='green', label='Rural')
]

plt.legend(handles=legend_patches, title="Area Type", loc="upper right")

# Labels
plt.title("Accident Hotspots by Area Type (Urban vs Rural)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.show()

