import minitorch as mt
import minitorch.nn as nn
import numpy as np

np.random.seed(42)

X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

X_tensor = mt.Tensor(X, requires_grad=False)
y_tensor = mt.Tensor(y, requires_grad=False)

model = nn.Sequential(
  nn.Linear(2, 10),
  nn.ReLU(),
  nn.Linear(10, 1),
  nn.Sigmoid()
)

optimizer = mt.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

print("Training simple binary classification model...")
for epoch in range(100):
  pred = model(X_tensor)
  
  loss = loss_fn(pred, y_tensor)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if (epoch + 1) % 20 == 0:
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nFinal predictions (first 10):")
with mt.Tensor(np.array([False]), requires_grad=False):
  final_pred = model(X_tensor[:10])
  print("Predictions:", final_pred.numpy().flatten())
  print("True labels:", y[:10].flatten())
