import minitorch as mt
import minitorch.nn as nn
import numpy as np

np.random.seed(42)

batch_size = 4
in_channels = 3
height, width = 8, 8

X = np.random.randn(batch_size, in_channels, height, width)
y = np.random.randint(0, 10, size=(batch_size,))

X_tensor = mt.Tensor(X, requires_grad=False)
y_onehot = np.eye(10)[y]
y_tensor = mt.Tensor(y_onehot, requires_grad=False)

class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(16 * 2 * 2, 32)
    self.fc2 = nn.Linear(32, 10)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x

model = SimpleCNN()
optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Training simple CNN...")
for epoch in range(50):
  pred = model(X_tensor)
  
  loss = loss_fn(pred, y_tensor)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if (epoch + 1) % 10 == 0:
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nModel architecture:")
print(f"Total parameters: {sum(p.data.size for p in model.parameters())}")
