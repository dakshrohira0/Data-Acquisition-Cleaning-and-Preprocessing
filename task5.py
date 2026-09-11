import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Load the cleaned dataset
df = pd.read_csv('cleaned_housing_dataset.csv')
print('Data shape:', df.shape)

# 2. One-hot encode categorical features (Neighborhood and Property Type)
df_ml = pd.get_dummies(
    df, columns=['Neighborhood', 'Property_Type'], drop_first=True
)

# 3. Select features (X) and target (y)
target = 'Price'
drop_cols = ['Property_ID', 'Price', 'Log_Price', 'Price_Per_Sqft']
feature_cols = [c for c in df_ml.columns if c not in drop_cols]

X = df_ml[feature_cols].values
y = df_ml[target].values.reshape(-1, 1)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Scale features using StandardScaler (crucial for neural networks)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert arrays to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)


# 5. Define Neural Network Architecture (MLP)
class SimpleHousingNN(nn.Module):

  def __init__(self, num_features):
    super(SimpleHousingNN, self).__init__()
    self.fc1 = nn.Linear(num_features, 64)  # Layer 1: 64 neurons
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(64, 32)  # Layer 2: 32 neurons
    self.relu2 = nn.ReLU()
    self.dropout = nn.Dropout(0.2)  # Dropout to prevent overfitting
    self.out = nn.Linear(32, 1)  # Output: 1 predicted price

  def forward(self, x):
    x = self.relu1(self.fc1(x))
    x = self.relu2(self.fc2(x))
    x = self.dropout(x)
    x = self.out(x)
    return x


model = SimpleHousingNN(num_features=X_train.shape[1])
print(model)

# Loss function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. Training Loop (100 Epochs)
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
  model.train()
  optimizer.zero_grad()  # Reset gradients

  # Forward pass
  predictions = model(X_train_t)
  loss = criterion(predictions, y_train_t)

  # Backward pass
  loss.backward()
  optimizer.step()

  # Validation loss check
  model.eval()
  with torch.no_grad():
    val_preds = model(X_test_t)
    val_loss = criterion(val_preds, y_test_t)

  train_losses.append(loss.item())
  test_losses.append(val_loss.item())

  if (epoch + 1) % 20 == 0:
    print(
        f'Epoch [{epoch+1}/{epochs}] - Train Loss: {loss.item():.4f} | Val'
        f' Loss: {val_loss.item():.4f}'
    )

# 7. Evaluate on Test Data (Convert back to real dollar scale)
model.eval()
with torch.no_grad():
  pred_scaled = model(X_test_t).numpy()
  pred_dollars = scaler_y.inverse_transform(pred_scaled)

mae = mean_absolute_error(y_test, pred_dollars)
rmse = np.sqrt(mean_squared_error(y_test, pred_dollars))
r2 = r2_score(y_test, pred_dollars)

print('\n--- Neural Network Test Results ---')
print(f'MAE  (Average Error):  ${mae:,.2f}')
print(f'RMSE (Root Error):     ${rmse:,.2f}')
print(f'R2 Score:              {r2:.4f}')

# 8. Plot Learning Curve (Loss vs Epochs)
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(
    range(1, epochs + 1),
    test_losses,
    label='Validation Loss',
    color='red',
    linestyle='--',
)
plt.title('Training & Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Plot Actual vs Predicted Prices
plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred_dollars, alpha=0.5, color='purple')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label='Ideal Fit',
)
plt.title('Neural Network: Actual vs Predicted')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.tight_layout()
plt.show()
