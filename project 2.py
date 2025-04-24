#eda(average vs peak water usage trends)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
df = pd.read_csv("water consumption project.csv")
print(f"duplicate rows: {df.duplicated().sum()}" )
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.date
hourly_usage = df.groupby('Hour')['Water_Usage_Liters'].agg(['mean', 'max'])
plt.figure(figsize=(10, 6))
sns.lineplot(data=hourly_usage, x=hourly_usage.index, y='mean', label='Average Usage')
sns.lineplot(data=hourly_usage, x=hourly_usage.index, y='max', label='Peak Usage')
plt.title('Average vs Peak Water Usage Trends')
plt.xlabel('Hour of the Day')
plt.ylabel('Water Usage (liters)')
plt.legend()
plt.show()


#eda (abnormal spikes)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
df = pd.read_csv("water consumption project.csv")
print(f"duplicate rows: {df.duplicated().sum()}" )
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.date
hourly_usage = df.groupby('Hour')['Water_Usage_Liters'].agg(['mean', 'max'])
hourly_usage['zscore_mean'] = zscore(hourly_usage['mean'])
hourly_usage['zscore_max'] = zscore(hourly_usage['max'])
abnormal_spikes_mean = hourly_usage[hourly_usage['zscore_mean'].abs() > 1]
abnormal_spikes_max = hourly_usage[hourly_usage['zscore_max'].abs() > 1]
plt.figure(figsize=(10, 6))
plt.scatter(abnormal_spikes_mean.index, abnormal_spikes_mean['mean'], color='blue', label='Abnormal Mean Spike')
plt.scatter(abnormal_spikes_max.index, abnormal_spikes_max['max'], color='red', label='Abnormal Max Spike')
plt.title('Abnormal Spikes')
plt.xlabel('Hour of the Day')
plt.ylabel('Water Usage (liters)')
plt.legend()
plt.show()




#linear regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
df = pd.read_csv("water consumption project.csv")
df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns)
if 'Weather_Condition' in df.columns:
    df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)
else:
    print("Warning: 'Weather_Condition' column not found!")
features = ['Water_Usage_Liters', 'Pressure_Bar']
features += [col for col in df.columns if "Weather_Condition_" in col]
target = 'Water_Usage_Liters'
df = df.dropna(subset=[target])
X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")
r2=r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")
threshold = 1.5 * rmse
df_test = df.iloc[y_test.index].copy()
df_test["Predicted Usage"] = y_pred
df_test["Leak Flag"] = (df_test["Water_Usage_Liters"] > df_test["Predicted Usage"] + threshold).astype(int)
print(df_test[["Water_Usage_Liters", "Predicted Usage", "Leak Flag"]])