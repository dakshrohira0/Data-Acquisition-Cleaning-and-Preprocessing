import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style='whitegrid')

# 1. Load the cleaned dataset
df = pd.read_csv('cleaned_housing_dataset.csv')
print('Data Shape:', df.shape)
print('\nSummary Statistics:\n', df.describe())

# 2. Chart 1: Univariate Distributions (Price and Square Footage)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df['Price'], kde=True, ax=axes[0], color='#2b6cb0', bins=25)
axes[0].set_title('Distribution of Property Listing Price')
axes[0].set_xlabel('Listing Price ($)')
axes[0].set_ylabel('Property Count')

sns.histplot(
    df['Square_Footage'], kde=True, ax=axes[1], color='#38a169', bins=25
)
axes[1].set_title('Distribution of Living Area (Square Footage)')
axes[1].set_xlabel('Square Footage (sq ft)')
axes[1].set_ylabel('Property Count')
plt.tight_layout()
plt.show()

# 3. Chart 2: Price Across Neighborhoods and Property Types (Boxplots)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
sns.boxplot(x='Neighborhood', y='Price', data=df, ax=axes[0], palette='Blues_r')
axes[0].set_title('Price Distribution by Neighborhood')
axes[0].tick_params(axis='x', rotation=30)

sns.boxplot(
    x='Property_Type', y='Price', data=df, ax=axes[1], palette='Greens_r'
)
axes[1].set_title('Price Distribution by Property Type')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()

# 4. Chart 3: Scatter Plot - Square Footage vs Price with Trendline
plt.figure(figsize=(8.5, 5))
sns.scatterplot(
    x='Square_Footage',
    y='Price',
    hue='Property_Type',
    data=df,
    alpha=0.7,
    palette='tab10',
)
sns.regplot(
    x='Square_Footage',
    y='Price',
    data=df,
    scatter=False,
    color='black',
    line_kws={'linestyle': '--', 'linewidth': 1.5},
)
plt.title('Square Footage vs Listing Price with Linear Trend')
plt.xlabel('Square Footage (sq ft)')
plt.ylabel('Listing Price ($)')
plt.legend(title='Property Type')
plt.tight_layout()
plt.show()

# 5. Chart 4: Average Price Per Square Foot by Neighborhood
plt.figure(figsize=(8.5, 4.5))
sns.barplot(
    x='Neighborhood',
    y='Price_Per_Sqft',
    data=df,
    palette='viridis',
    edgecolor='black',
)
plt.title('Average Price Per Square Foot Across Neighborhoods')
plt.xlabel('Neighborhood')
plt.ylabel('Average Price per Sq Ft ($)')
plt.tight_layout()
plt.show()

# 6. Chart 5: Correlation Heatmap
plt.figure(figsize=(7.5, 5.5))
numeric_cols = [
    'Bedrooms',
    'Bathrooms',
    'Square_Footage',
    'Property_Age',
    'Days_On_Market',
    'Price',
    'Price_Per_Sqft',
]
corr = df[numeric_cols].corr()
sns.heatmap(
    corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1
)
plt.title('Correlation Heatmap of Quantitative Features')
plt.tight_layout()
plt.show()
