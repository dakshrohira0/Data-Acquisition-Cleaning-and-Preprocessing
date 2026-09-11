import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# 1. Load the cleaned dataset
df = pd.read_csv('cleaned_housing_dataset.csv')
print('Data shape:', df.shape)

# 2. Convert categorical text columns to dummy numbers (One-Hot Encoding)
df_ml = pd.get_dummies(
    df, columns=['Neighborhood', 'Property_Type'], drop_first=True
)

# 3. Separate features (X) and target (y)
target = 'Price'
# Drop IDs and columns derived directly from Price to prevent data leakage
drop_columns = ['Property_ID', 'Price', 'Log_Price', 'Price_Per_Sqft']
feature_columns = [col for col in df_ml.columns if col not in drop_columns]

X = df_ml[feature_columns]
y = df_ml[target]

print('Features used (total', len(feature_columns), '):', feature_columns)

# 4. Split into Train and Test sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f'Training samples: {len(X_train)}, Testing samples: {len(X_test)}')

# 5. Train Linear Regression Model (Baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 6. Train Random Forest Regressor Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# 7. Evaluate both models on Test data
def print_scores(name, y_true, y_pred):
  mae = mean_absolute_error(y_true, y_pred)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  r2 = r2_score(y_true, y_pred)
  print(f'\n--- {name} Results ---')
  print(f'MAE  (Avg Error):  ${mae:,.2f}')
  print(f'RMSE (Root Error): ${rmse:,.2f}')
  print(f'R2 Score:          {r2:.4f}')


print_scores('Linear Regression', y_test, y_pred_lr)
print_scores('Random Forest', y_test, y_pred_rf)

# 8. Perform 5-Fold Cross Validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f'\nRandom Forest 5-Fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

# 9. Plot Actual vs Predicted Prices
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.5, label='Properties')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2,
    label='Ideal Line',
)
plt.title('Random Forest: Actual vs Predicted Price')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.tight_layout()
plt.show()

# 10. Feature Importance Bar Chart
importances = pd.Series(rf.feature_importances_, index=feature_columns)
importances.sort_values().plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Feature Importances in Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
