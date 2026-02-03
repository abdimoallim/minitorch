import sys
sys.path.insert(0, '/home/claude')
import minitorch as mt
import minitorch.nn as nn
import numpy as np

np.random.seed(42)

print("MiniTorch - Complete Training Pipeline Demo")
print("=" * 60)

print("\n1. Generating synthetic dataset (simulating MNIST)...")
num_samples = 1000
num_classes = 10
input_dim = 28 * 28

X_train = np.random.randn(num_samples, input_dim).astype(np.float32) * 0.1
y_train = np.random.randint(0, num_classes, size=num_samples)
y_train_onehot = np.eye(num_classes)[y_train]

X_test = np.random.randn(200, input_dim).astype(np.float32) * 0.1
y_test = np.random.randint(0, num_classes, size=200)
y_test_onehot = np.eye(num_classes)[y_test]

print(f"Training set: {X_train.shape}, {y_train_onehot.shape}")
print(f"Test set: {X_test.shape}, {y_test_onehot.shape}")

print("\n2. Building neural network model...")

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])
    self.bn1 = nn.BatchNorm1d(hidden_dims[0])
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.bn2 = nn.BatchNorm1d(hidden_dims[1])
    self.fc3 = nn.Linear(hidden_dims[1], output_dim)
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.relu(self.bn1(self.fc1(x)))
    x = self.dropout(x)
    x = self.relu(self.bn2(self.fc2(x)))
    x = self.dropout(x)
    x = self.fc3(x)
    return x

model = MLP(input_dim=input_dim, hidden_dims=[256, 128], output_dim=num_classes)

total_params = sum(p.data.size for p in model.parameters())
print(f"Model created with {total_params:,} parameters")
print("\nModel architecture:")
print("  Layer 1: Linear(784, 256) + BatchNorm + ReLU + Dropout(0.2)")
print("  Layer 2: Linear(256, 128) + BatchNorm + ReLU + Dropout(0.2)")
print("  Layer 3: Linear(128, 10)")

print("\n3. Setting up training...")
optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

batch_size = 32
num_epochs = 10

print(f"Optimizer: Adam (lr=0.001)")
print(f"Loss: CrossEntropyLoss")
print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}")

print("\n4. Training the model...")
print("-" * 60)

for epoch in range(num_epochs):
  model.train()
  epoch_loss = 0
  num_batches = 0
  
  indices = np.random.permutation(num_samples)
  X_shuffled = X_train[indices]
  y_shuffled = y_train_onehot[indices]
  
  for i in range(0, num_samples, batch_size):
    batch_X = X_shuffled[i:i+batch_size]
    batch_y = y_shuffled[i:i+batch_size]
    
    X_batch = mt.Tensor(batch_X, requires_grad=False)
    y_batch = mt.Tensor(batch_y, requires_grad=False)
    
    pred = model(X_batch)
    loss = loss_fn(pred, y_batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    epoch_loss += loss.item()
    num_batches += 1
  
  avg_loss = epoch_loss / num_batches
  
  if (epoch + 1) % 2 == 0:
    model.eval()
    X_test_tensor = mt.Tensor(X_test, requires_grad=False)
    y_test_tensor = mt.Tensor(y_test_onehot, requires_grad=False)
    
    test_pred = model(X_test_tensor)
    test_loss = loss_fn(test_pred, y_test_tensor)
    
    pred_classes = np.argmax(test_pred.data, axis=1)
    accuracy = (pred_classes == y_test).mean() * 100
    
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {avg_loss:.4f} | "
          f"Test Loss: {test_loss.item():.4f} | "
          f"Accuracy: {accuracy:.2f}%")

print("-" * 60)
print("\n5. Final evaluation...")

model.eval()
X_test_tensor = mt.Tensor(X_test, requires_grad=False)
final_pred = model(X_test_tensor)
pred_classes = np.argmax(final_pred.data, axis=1)

accuracy = (pred_classes == y_test).mean() * 100
print(f"Final Test Accuracy: {accuracy:.2f}%")

print("\nSample predictions (first 10):")
print("Predicted:", pred_classes[:10])
print("Actual:   ", y_test[:10])

print("\n6. Testing gradient flow...")
sample_input = mt.Tensor(X_test[0:1], requires_grad=True)
sample_output = model(sample_input)
sample_loss = sample_output.sum()
sample_loss.backward()

print(f"Input gradient shape: {sample_input.grad.shape}")
print(f"Input gradient norm: {np.linalg.norm(sample_input.grad.data):.4f}")
print("✓ Gradients successfully computed through the entire network")

print("\n" + "=" * 60)
print("Training pipeline completed successfully!")
print("=" * 60)

print("\nMiniTorch Features Demonstrated:")
print("  ✓ Tensor operations with autograd")
print("  ✓ Multi-layer neural networks")
print("  ✓ Batch normalization")
print("  ✓ Dropout regularization")
print("  ✓ Adam optimizer")
print("  ✓ Cross-entropy loss")
print("  ✓ Mini-batch training")
print("  ✓ Train/eval modes")
print("  ✓ Gradient computation")
