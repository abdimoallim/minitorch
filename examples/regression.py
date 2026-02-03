import minitorch as mt
import minitorch.nn as nn
import numpy as np

np.random.seed(42)

X = np.random.randn(200, 1)
y = 3 * X**2 + 2 * X + 1 + np.random.randn(200, 1) * 0.5

X_tensor = mt.Tensor(X, requires_grad=False)
y_tensor = mt.Tensor(y, requires_grad=False)

class MLPRegressor(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, 1)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = MLPRegressor()
optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Training regression model...")
for epoch in range(500):
  pred = model(X_tensor)
  
  loss = loss_fn(pred, y_tensor)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if (epoch + 1) % 100 == 0:
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nTesting on new data:")
X_test = np.array([[-2], [-1], [0], [1], [2]])
X_test_tensor = mt.Tensor(X_test, requires_grad=False)

model.eval()
predictions = model(X_test_tensor)

print("Input | Prediction | True Value")
for i, x_val in enumerate(X_test):
  true_val = 3 * x_val[0]**2 + 2 * x_val[0] + 1
  print(f"{x_val[0]:5.1f} | {predictions.numpy()[i, 0]:10.4f} | {true_val:10.4f}")
