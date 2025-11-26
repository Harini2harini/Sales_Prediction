import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import norm, chisquare
from statsmodels.stats.weightstats import ztest
# Load dataset
df = pd.read_csv("C:/harini/sales_data.csv")
# ==============================
# 1. TRANSFORMATION TECHNIQUES
# ==============================
print("\n--- TRANSFORMATION TECHNIQUES ---")
# Label Encoding categorical columns
encoder = LabelEncoder()
df['Customer_Gender_enc'] = encoder.fit_transform(df['Customer_Gender'])
print("Label Encoding applied on 'Customer_Gender' -> New column: 'Customer_Gender_enc'")
# Scaling numerical columns
scaler = StandardScaler()
df['Revenue_scaled'] = scaler.fit_transform(df[['Revenue']])
print("StandardScaler applied on 'Revenue' -> New column: 'Revenue_scaled'")
minmax = MinMaxScaler()
df['Profit_norm'] = minmax.fit_transform(df[['Profit']])
print("MinMaxScaler applied on 'Profit' -> New column: 'Profit_norm'")
# Log transformation (handle zeroes)
df['Revenue_log'] = np.log1p(df['Revenue'])
print("Log Transformation applied on 'Revenue' -> New column: 'Revenue_log'")
# ==============================
# 2. STATISTICAL ANALYSIS
# ==============================
print("\n--- STATISTICAL ANALYSIS ---")
# Gaussian Distribution Fit on Revenue
mu, sigma = norm.fit(df['Revenue'])
print(f"Gaussian Fit -> Mean: {mu:.2f}, StdDev: {sigma:.2f}")
# Chi-Square Test on Gender distribution
obs_counts = df['Customer_Gender'].value_counts()
chi_stat, chi_p = chisquare(obs_counts)
print(f"Chi-Square Test (Gender Distribution) -> stat={chi_stat:.2f}, p-value={chi_p:.4f}")
# Z-test: check if mean profit differs from 250
z_stat, z_p = ztest(df['Profit'], value=250)
print(f"Z-Test (Profit vs Hypothesized Mean=250) -> stat={z_stat:.2f}, p-value={z_p:.4f}")
# ==============================
# 3. VISUALIZATIONS
# ==============================
print("\n--- VISUALIZATIONS ---")
print("1. Lineplot -> Average Revenue by Year")
print("2. Barplot -> Average Revenue by Country")
print("3. Area Plot -> Cumulative Profit Over Years")
print("4. Stackplot -> Order Quantity by Gender across Years")
print("5. Scatterplot -> Profit vs Revenue")
print("6. Pie Chart -> Product Category Distribution")
print("7. Table Chart -> Summary Statistics of Revenue, Profit, Order Quantity")
print("8. Polar Chart -> Average Revenue across Months")
print("9. Box & Whisker Plot -> Profit distribution by Gender")
print("10. Heatmap -> Correlation between numerical features")
plt.figure(figsize=(20, 25))
# Lineplot: Revenue trend by Year
plt.subplot(5,2,1)
sns.lineplot(x="Year", y="Revenue", data=df, estimator='mean')
plt.title("Average Revenue by Year")
# Barplot: Avg Revenue by Country
plt.subplot(5,2,2)
sns.barplot(x="Country", y="Revenue", data=df, estimator=np.mean)
plt.title("Average Revenue by Country")
# Area Plot: Cumulative Profit
plt.subplot(5,2,3)
df.groupby('Year')['Profit'].sum().cumsum().plot.area(alpha=0.6)
plt.title("Cumulative Profit Over Years")
# Stackplot: Order Quantity vs Year by Gender
plt.subplot(5,2,4)
years = sorted(df['Year'].unique())
male = df[df['Customer_Gender']=="M"].groupby('Year')['Order_Quantity'].sum()
female = df[df['Customer_Gender']=="F"].groupby('Year')['Order_Quantity'].sum()
plt.stackplot(years, male, female, labels=["Male","Female"], alpha=0.7)
plt.legend(loc='upper left')
plt.title("Stackplot of Order Quantity by Gender")
# Scatterplot: Revenue vs Profit
plt.subplot(5,2,5)
plt.scatter(df['Profit'], df['Revenue'], alpha=0.3)
plt.xlabel("Profit"); plt.ylabel("Revenue")
plt.title("Scatterplot: Profit vs Revenue")
# Pie Chart: Product Category Distribution
plt.subplot(5,2,6)
df['Product_Category'].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Product Category Distribution")
# Table Chart: Summary Stats
plt.subplot(5,2,7)
plt.axis('off')
summary = df[['Revenue','Profit','Order_Quantity']].describe().round(2)
plt.table(cellText=summary.values, colLabels=summary.columns,
rowLabels=summary.index, loc='center')
plt.title("Summary Statistics Table")
# Polar Chart: Revenue across Months
plt.subplot(5,2,8, polar=True)
month_map = {m:i for i,m in enumerate(df['Month'].unique())}
month_revenue = df.groupby('Month')['Revenue'].mean()
angles = np.linspace(0, 2*np.pi, len(month_revenue), endpoint=False)
plt.polar(angles, month_revenue)
plt.fill(angles, month_revenue, alpha=0.3)
plt.title("Average Revenue by Month (Polar Chart)")
# Box & Whisker: Profit by Gender
plt.subplot(5,2,9)
sns.boxplot(x="Customer_Gender", y="Profit", data=df)
plt.title("Profit Distribution by Gender")
# Heatmap: Correlation Matrix
plt.subplot(5,2,10)
sns.heatmap(df[['Revenue','Profit','Order_Quantity','Unit_Cost','Unit_Price']].corr(),
annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
# ==============================
# 11. Gaussian Distribution Curve
# ==============================
print("11. Gaussian Distribution Curve -> Histogram of Revenue with Gaussian Fit")
plt.figure(figsize=(7,5))
sns.histplot(df['Revenue'].dropna(), kde=False, bins=30, stat='density', color="skyblue")
# Fit normal distribution
mu, sigma = norm.fit(df['Revenue'].dropna())
x = np.linspace(df['Revenue'].min(), df['Revenue'].max(), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'N({mu:.2f}, {sigma:.2f}²)')
plt.title("Gaussian Distribution Curve for Revenue")
plt.xlabel("Revenue")
plt.ylabel("Density")
plt.legend()
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
# Select features and target
features = ['Profit', 'Order_Quantity', 'Unit_Cost', 'Unit_Price']
X = df[features]
y = df['Revenue']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
# Evaluation
print("\n--- Regression on Revenue ---")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
from scipy.stats import zscore
# IQR Method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))]
    return outliers
# Z-score Method
def detect_outliers_zscore(data, column, threshold=3):
    z_scores = zscore(data[column])
    outliers = data[np.abs(z_scores) > threshold]
    return outliers
# Apply on Profit
iqr_outliers = detect_outliers_iqr(df, 'Profit')
zscore_outliers = detect_outliers_zscore(df, 'Profit')
print("\n--- Outlier Detection ---")
print(f"IQR Method: {len(iqr_outliers)} outliers detected in 'Profit'")
print(f"Z-Score Method: {len(zscore_outliers)} outliers detected in 'Profit'")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Clustering on selected numerical features
cluster_features = df[['Revenue', 'Profit', 'Order_Quantity']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)
# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Customer_Cluster'] = kmeans.fit_predict(scaled_features)
print("\n--- KMeans Clustering ---")
print(df['Customer_Cluster'].value_counts())
# Optional: Visualize clusters
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
sns.scatterplot(x='Revenue', y='Profit', hue='Customer_Cluster', data=df, palette='Set2')
plt.title("Customer Segmentation based on Revenue and Profit")
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
# Ensure Month is ordered and combine with Year to create time series index
df['Month'] = pd.Categorical(df['Month'], categories=[
'January','February','March','April','May','June',
'July','August','September','October','November','December'], ordered=True)
# Create datetime index
# Drop NaT and sort
ts_data = df.dropna(subset=['Date']).sort_values('Date')
monthly_revenue = ts_data.groupby('Date')['Revenue'].sum()
# Decomposition
decomposition = seasonal_decompose(monthly_revenue, model='additive', period=12)
print("\n--- Time Series Decomposition of Monthly Revenue ---")
decomposition.plot()
plt.tight_layout()
plt.show()
# ==============================
# 4. DATA AGGREGATION
# ==============================
print("\n--- DATA AGGREGATION ---")
# Aggregation by Year
agg_year = df.groupby('Year')[['Revenue', 'Profit', 'Order_Quantity']].agg(['sum',
'mean']).round(2)
print("\nTotal and Average Revenue, Profit, and Orders by Year:")
print(agg_year)
# Aggregation by Country
agg_country = df.groupby('Country')[['Revenue', 'Profit']].agg(['sum', 'mean']).round(2)
print("\nTotal and Average Revenue & Profit by Country:")
print(agg_country)
# Aggregation by Product Category
agg_category = df.groupby('Product_Category')[['Revenue', 'Profit','Order_Quantity']].agg(['sum', 'mean']).round(2)
print("\nProduct Category-wise Aggregated Data:")
print(agg_category)
# Aggregation by Gender
agg_gender = df.groupby('Customer_Gender')[['Revenue', 'Profit']].agg(['sum',
'mean']).round(2)
print("\nGender-wise Aggregated Revenue & Profit:")
print(agg_gender)