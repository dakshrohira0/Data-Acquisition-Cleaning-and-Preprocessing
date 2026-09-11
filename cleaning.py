import numpy as np
import pandas as pd

# 1. Load the dataset
df = pd.read_csv('raw_housing_data.csv')

# Check basic info
print('Shape before cleaning:', df.shape)
print(df.head())
print(df.isnull().sum())

# 2. Remove duplicate rows
df = df.drop_duplicates()
print('Shape after removing duplicates:', df.shape)

# 3. Clean categorical text columns (fix lowercase and spaces)
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


# 4. Clean the Price column (remove $ and commas)
def fix_price(p):
  if pd.isna(p):
    return np.nan
  p = str(p)
  if p in ['N/A', 'Unknown', 'None', 'nan']:
    return np.nan
  # remove special chars
  p = p.replace('$', '').replace(',', '').strip()
  try:
    val = float(p)
    if val > 10000:  # prices shouldn't be too small
      return val
    else:
      return np.nan
  except:
    return np.nan


df['Price'] = df['Listing_Price'].apply(fix_price)

# 5. Fix invalid / erroneous values
# Square footage can't be negative or too huge
df.loc[df['Square_Footage'] < 0, 'Square_Footage'] = abs(df['Square_Footage'])
df.loc[df['Square_Footage'] > 15000, 'Square_Footage'] = np.nan
df.loc[df['Square_Footage'] < 200, 'Square_Footage'] = np.nan

# Bedrooms can't be 0, negative, or 99
df.loc[df['Bedrooms'] <= 0, 'Bedrooms'] = np.nan
df.loc[df['Bedrooms'] > 15, 'Bedrooms'] = np.nan

# Year built cannot be in the future or unrealistic past
df.loc[df['Year_Built'] < 1880, 'Year_Built'] = np.nan
df.loc[df['Year_Built'] > 2026, 'Year_Built'] = np.nan

# 6. Fill missing values (Imputation)
# Fill categorical with mode
df['Neighborhood'] = df['Neighborhood'].fillna(df['Neighborhood'].mode()[0])
df['Property_Type'] = df['Property_Type'].fillna(df['Property_Type'].mode()[0])

# Fill numerical columns with median
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())
df['Square_Footage'] = df['Square_Footage'].fillna(df['Square_Footage'].median())
df['Year_Built'] = df['Year_Built'].fillna(df['Year_Built'].median())
df['Days_On_Market'] = df['Days_On_Market'].fillna(df['Days_On_Market'].median())
df['Price'] = df['Price'].fillna(df['Price'].median())

# 7. Outlier treatment for Price using simple IQR method
q1 = df['Price'].quantile(0.25)
q3 = df['Price'].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr

df['Price'] = df['Price'].clip(lower=lower, upper=upper)

# 8. Create new features
df['Property_Age'] = 2026 - df['Year_Built']
df['Price_Per_Sqft'] = round(df['Price'] / df['Square_Footage'], 2)

# 9. Keep final columns and export to CSV
final_columns = [
    'Property_ID',
    'Neighborhood',
    'Property_Type',
    'Bedrooms',
    'Bathrooms',
    'Square_Footage',
    'Property_Age',
    'Days_On_Market',
    'Price',
    'Price_Per_Sqft',
]

df_cleaned = df[final_columns]
df_cleaned.to_csv('cleaned_housing_dataset.csv', index=False)

print('Cleaning complete!')
print('Final missing values:\n', df_cleaned.isnull().sum())
print('Final shape:', df_cleaned.shape)
