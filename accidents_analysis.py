import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load dataset
df = pd.read_csv("accident.csv", encoding='latin1')

#HANDLING NULL VALUES

df_cleaned = df.copy()

# Drop column with too many missing values, 
df_cleaned = df.drop(columns=["TWAY_ID2"], errors='ignore')

# Create derived column, handling potential division by zero
df_cleaned['VEH_PER_PERSON'] = df_cleaned['VE_TOTAL'] / (df_cleaned['PERSONS'] + 0.1)

# Handle invalid/missing time data (99 = unknown)
df_cleaned['TIME_DELAY'] = (df_cleaned["ARR_HOUR"] - df_cleaned["HOUR"]).replace(99, np.nan)

sns.set(style="whitegrid")

'''
# 3. OBJECTIVE 1: Temporal Fatality Patterns (Hour of Day)
# hour data must be between 0 to 24 hours

df_cleaned = df[df["HOUR"].between(0, 23)]

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


# OBJECTIVE 2: Interaction of Weather and Light Conditions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 🔍 Objective 3: Spatial Clustering of High-Fatality Crash Zones
# Using LATITUDE, LONGITUD, and RUR_URBNAME (Urban/Rural)
# ------------------------------------------------------

# 1. Filter valid GPS coordinates
df_location = df_cleaned[
    df_cleaned["LATITUDE"].between(-90, 90) & 
    df_cleaned["LONGITUD"].between(-180, 180)
]

# 2.  Remove rows with NaN in urban/rural classification
df_location = df_location[df_location["RUR_URBNAME"].notnull()]

# 3. Plot: Density of fatal crashes by location (basic heatmap view)
plt.figure(figsize=(10, 8))
sns.kdeplot(
    data=df_location, 
    x="LONGITUD", 
    y="LATITUDE", 
    cmap="Reds", 
    shade=True, 
    bw_adjust=0.5,
    thresh=0.05
)

plt.title("📍 Spatial Density of Fatal Crashes (U.S.)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# Group by urban vs rural areas
# To compare number of fatalities based on area type
# ---------------------------------------------
urban_rural_stats = df_location.groupby("RUR_URBNAME")["FATALS"].sum().reset_index().sort_values(by="FATALS", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=urban_rural_stats, x="RUR_URBNAME", y="FATALS", palette="mako")
plt.title("Fatalities by Urban/Rural Classification")
plt.xlabel("Area Type")
plt.ylabel("Total Fatalities")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()




