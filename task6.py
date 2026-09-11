import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Set general plot styling
sns.set_theme(style='whitegrid')

# ==========================================
# STEP 1: LOAD & CLEAN THE DATA
# ==========================================
print('--- STEP 1: DATA CLEANING ---')
df = pd.read_csv('raw_housing_data.csv')
print('Raw shape:', df.shape)

# Drop duplicate rows
df = df.drop_duplicates()

# Clean categorical text columns (fix spaces & casing)
df['Neighborhood'] = df['Neighborhood'].astype(str).str.strip().str.title()
df['Neighborhood'] = df['Neighborhood'].replace(
    {'Westbay': 'West Bay', 'Nan': np.nan, 'None': np.nan}
)

df['Property_Type'] = df['Property_Type'].astype(str).str.strip().str.title()
df['Property_Type'] = df['Property_Type'].replace({
    'Sfh': 'Single Family',
    'Town House': 'Townhouse',
    'Nan': np.nan,
    'None': np.nan,
})


# Clean and parse the Price column
def clean_price(val):
  if pd.isna(val):
    return np.nan
  s = str(val).strip().lower()
  if s in ['n/a', 'unknown', 'none', 'nan', '']:
    return np.nan
  cleaned_str = str(val).replace('$', '').replace(',', '').strip()
  try:
    num = float(cleaned_str)
    return num if num > 10000 else np.nan
  except:
    return np.nan


df['Price'] = df['Listing_Price'].apply(clean_price)

# Fix erroneous bounds (negative or extreme values)
df.loc[df['Square_Footage'] < 0, 'Square_Footage'] = abs(df['Square_Footage'])
df.loc[
    (df['Square_Footage'] > 15000) | (df['Square_Footage'] < 200),
    'Square_Footage',
] = np.nan
df.loc[(df['Bedrooms'] <= 0) | (df['Bedrooms'] > 15), 'Bedrooms'] = np.nan
df.loc[(df['Year_Built'] < 1880) | (df['Year_Built'] > 2026), 'Year_Built'] = (
    np.nan
)

# Fill missing values (Mode for categorical, Median for numerical)
df['Neighborhood'] = df['Neighborhood'].fillna(df['Neighborhood'].mode()[0])
df['Property_Type'] = df['Property_Type'].fillna(df['Property_Type'].mode()[0])
for col in [
    'Bedrooms',
    'Bathrooms',
    'Square_Footage',
    'Year_Built',
    'Days_On_Market',
    'Price',
]:
  df[col] = df[col].fillna(df[col].median())

# Outlier treatment using simple IQR method
q1 = df['Price'].quantile(0.25)
q3 = df['Price'].quantile(0.75)
iqr = q3 - q1
df['Price'] = df['Price'].clip(lower=max(0, q1 - 1.5 * iqr), upper=q3 + 1.5 * iqr)

# Create engineered features
df['Property_Age'] = 2026 - df['Year_Built']
df['Price_Per_Sqft'] = (df['Price'] / df['Square_Footage']).round(2)
print('Cleaned dataset shape:', df.shape)

# ==========================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print('\n--- STEP 2: EXPLORATORY DATA ANALYSIS ---')
eda_cols = [
    'Bedrooms',
    'Bathrooms',
    'Square_Footage',
    'Property_Age',
    'Price',
    'Price_Per_Sqft',
]
print('Correlation with Price:\n', df[eda_cols].corr()['Price'])

# ==========================================
# STEP 3: UNSUPERVISED LEARNING (CLUSTERING)
# ==========================================
print('\n--- STEP 3: UNSUPERVISED CLUSTERING ---')
cluster_features = ['Square_Footage', 'Price', 'Price_Per_Sqft', 'Property_Age']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cluster_features])

# Fit K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_names = {
    0: 'Modern Starter Homes',
    1: 'Premium Estates',
    2: 'Mature Historic Homes',
}
df['Cluster_Name'] = df['Cluster'].map(cluster_names)
print('Cluster Averages:\n', df.groupby('Cluster_Name')[cluster_features].mean())

# ==========================================
# STEP 4: SUPERVISED LEARNING (REGRESSION)
# ==========================================
print('\n--- STEP 4: SUPERVISED PREDICTIVE MODELING ---')
# One-hot encode categories for machine learning
df_ml = pd.get_dummies(
    df, columns=['Neighborhood', 'Property_Type'], drop_first=True
)

# Avoid target leakage: drop IDs, Price, and columns created from Price
drop_cols = [
  'Property_ID',
  'Price',
  'Listing_Price',
  'Log_Price',
  'Price_Per_Sqft',
  'Cluster',
  'Cluster_Name',
]
feature_cols = [c for c in df_ml.columns if c not in drop_cols]

X = df_ml[feature_cols]
y = df_ml['Price']

# Split data into 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train Baseline Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# Evaluate performance
print('\nModel Evaluation Results on Test Data:')
print(
    f'Linear Regression -> MAE: ${mean_absolute_error(y_test, pred_lr):,.2f},'
    f' R2: {r2_score(y_test, pred_lr):.4f}'
)
print(
    f'Random Forest     -> MAE: ${mean_absolute_error(y_test, pred_rf):,.2f},'
    f' R2: {r2_score(y_test, pred_rf):.4f}'
)

# 5-fold cross validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f'Random Forest 5-Fold CV R2: {cv_scores.mean():.4f}')

# ==========================================
# STEP 5: SAVE FINAL RESULTS
# ==========================================
df.to_csv('capstone_final_dataset.csv', index=False)
print('\n[COMPLETED] Pipeline finished. Saved to capstone_final_dataset.csv')
