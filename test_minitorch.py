import sys
sys.path.insert(0, '/home/claude')

import minitorch as mt
import minitorch.nn as nn
import numpy as np

def test_basic_ops():
  print("Testing basic operations...")
  x = mt.Tensor([[1, 2], [3, 4]], requires_grad=True)
  y = mt.Tensor([[5, 6], [7, 8]], requires_grad=True)
  
  z = x + y
  assert z.data.shape == (2, 2)
  
  w = x @ y
  assert w.data.shape == (2, 2)
  
  loss = w.sum()
  loss.backward()
  
  assert x.grad is not None
  assert y.grad is not None
  print("✓ Basic operations passed")

def test_activations():
  print("Testing activations...")
  x = mt.Tensor([[-1, 0, 1, 2]], requires_grad=True)
  
  relu_out = x.relu()
  assert np.allclose(relu_out.data, [[0, 0, 1, 2]])
  
  sigmoid_out = x.sigmoid()
  assert sigmoid_out.data.shape == x.shape
  
  tanh_out = x.tanh()
  assert tanh_out.data.shape == x.shape
  
  print("✓ Activations passed")

def test_reductions():
  print("Testing reductions...")
  x = mt.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
  
  total = x.sum()
  assert np.isclose(total.data, 21)
  
  mean_val = x.mean()
  assert np.isclose(mean_val.data, 3.5)
  
  max_val = x.max()
  assert np.isclose(max_val.data, 6)
  
  sum_dim0 = x.sum(dim=0)
  assert sum_dim0.shape == (3,)
  
  print("✓ Reductions passed")

def test_reshape():
  print("Testing reshape and transpose...")
  x = mt.Tensor([[1, 2, 3], [4, 5, 6]])
  
  reshaped = x.reshape(3, 2)
  assert reshaped.shape == (3, 2)
  
  transposed = x.transpose()
  assert transposed.shape == (3, 2)
  
  print("✓ Reshape and transpose passed")

def test_linear_layer():
  print("Testing linear layer...")
  layer = nn.Linear(5, 3)
  x = mt.Tensor(np.random.randn(2, 5))
  
  out = layer(x)
  assert out.shape == (2, 3)
  
  params = layer.parameters()
  assert len(params) == 2
  
  print("✓ Linear layer passed")

def test_sequential():
  print("Testing sequential model...")
  model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
  )
  
  x = mt.Tensor(np.random.randn(3, 10))
  out = model(x)
  assert out.shape == (3, 5)
  
  params = model.parameters()
  assert len(params) == 4
  
  print("✓ Sequential model passed")

def test_optimizer():
  print("Testing optimizer...")
  x = mt.Tensor([[1, 2], [3, 4]], requires_grad=True)
  optimizer = mt.optim.SGD([x], lr=0.1)
  
  y = (x ** 2).sum()
  y.backward()
  
  old_data = x.data.copy()
  optimizer.step()
  
  assert not np.allclose(x.data, old_data)
  print("✓ Optimizer passed")

def test_training_loop():
  print("Testing simple training loop...")
  np.random.seed(42)
  
  X = np.random.randn(50, 2)
  y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
  
  X_tensor = mt.Tensor(X, requires_grad=False)
  y_tensor = mt.Tensor(y, requires_grad=False)
  
  model = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
  )
  
  optimizer = mt.optim.Adam(model.parameters(), lr=0.01)
  loss_fn = nn.BCELoss()
  
  initial_loss = None
  for epoch in range(20):
    pred = model(X_tensor)
    loss = loss_fn(pred, y_tensor)
    
    if initial_loss is None:
      initial_loss = loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  final_loss = loss.item()
  assert final_loss < initial_loss, "Loss should decrease during training"
  
  print(f"✓ Training loop passed (loss: {initial_loss:.4f} → {final_loss:.4f})")

if __name__ == "__main__":
  print("=" * 50)
  print("Running MiniTorch Tests")
  print("=" * 50)
  
  try:
    test_basic_ops()
    test_activations()
    test_reductions()
    test_reshape()
    test_linear_layer()
    test_sequential()
    test_optimizer()
    test_training_loop()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
  except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
