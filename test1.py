"""import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('traffic_accident_data.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.columns.to_list())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('traffic_accident_data.csv')
print(df)


#eda1 heatmap

df['Date-time'] = pd.to_datetime(df['Date-time'])
df['Date'] = df['Date-time'].dt.date
df['Hour'] = df['Date-time'].dt.hour
heatmap_data = df.pivot_table(index='Location', columns='Hour', values='Accident ID', aggfunc='count').fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5)
plt.title("Heatmap of Accident Hotspots by Hour and Location")
plt.xlabel("Hour of the Day")
plt.ylabel("Location")
plt.tight_layout()
plt.show()



#eda2 bar chart

severity_counts = df['Severity'].value_counts()
plt.figure(figsize=(8, 6))
severity_counts.plot(kind='bar', color=['lightcoral', 'gold', 'lightgreen'])
plt.title("Accident Counts by Severity")
plt.xlabel("Severity")
plt.ylabel("Number of Accidents")
plt.tight_layout()
weather_counts = df['Weather Condition'].value_counts()
plt.figure(figsize=(8, 6))
weather_counts.plot(kind='bar', color=['skyblue', 'gray', 'lightgreen'])
plt.title("Accident Counts by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()



#eda3 line plot

accidents_per_day = df.groupby('Date')['Accident ID'].count().reset_index(name='Accident Count')
plt.figure(figsize=(10, 6))
plt.plot(accidents_per_day['Date'], accidents_per_day['Accident Count'], marker='o', color='orange', linestyle='-')
plt.title("Accident Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Accidents")
plt.grid(axis='y')
plt.tight_layout()
plt.show()