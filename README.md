### minitorch

A minimal PyTorch-like deep learning library implemented in Python with NumPy and CUDA support.

### Features

- Autograd engine with gradient computation (incomplete)
- Element-wise ops, matrix multiplication, reductions, reshaping
- Linear layers, Conv2D, pooling, batch norm, dropout
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, Cross Entropy, Binary Cross Entropy, L1 Loss)
- Optimizers (SGD, Adam, RMSprop, Adagrad)
- CUDA kernels for GPU operations (optional)

### Installation

```bash
pip install numpy
pip install -e .
```

For CUDA support:
```bash
pip install cupy-cuda11x
```

### Quick Start

#### Basic Tensor Operations

```python
import minitorch as mt

x = mt.Tensor([[1, 2], [3, 4]], requires_grad=True)
y = mt.Tensor([[5, 6], [7, 8]], requires_grad=True)

z = x @ y
loss = z.sum()
loss.backward()

print(x.grad)
```

#### Building a Neural Network

```python
import minitorch as mt
import minitorch.nn as nn

model = nn.Sequential(
  nn.Linear(784, 128),
  nn.ReLU(),
  nn.Dropout(0.2),
  nn.Linear(128, 10),
  nn.Softmax(dim=-1)
)

optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
  pred = model(X)
  loss = loss_fn(pred, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```

#### Convolutional Neural Networks

```python
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.pool = nn.MaxPool2d(2)
    self.fc = nn.Linear(64 * 5 * 5, 10)
  
  def forward(self, x):
    x = self.pool(self.conv1(x).relu())
    x = self.pool(self.conv2(x).relu())
    x = x.reshape(x.shape[0], -1)
    return self.fc(x)
```

#### CUDA Support

```python
x = mt.Tensor([[1, 2], [3, 4]], requires_grad=True, device='cuda')
y = mt.Tensor([[5, 6], [7, 8]], requires_grad=True, device='cuda')

z = x @ y
```

Or move existing tensors:

```python
x = mt.Tensor([[1, 2], [3, 4]], requires_grad=True)
x = x.cuda()

model = CNN()
model.cuda()
```

### API Reference

#### Tensor Creation

- `mt.Tensor(data, requires_grad=False, device='cpu')`
- `mt.zeros(*shape)`
- `mt.ones(*shape)`
- `mt.randn(*shape)`
- `mt.rand(*shape)`
- `mt.arange(start, end, step)`
- `mt.eye(n, m)`

#### Tensor Operations

- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Matrix ops: `@` (matmul), `.t()` (transpose)
- Reductions: `.sum()`, `.mean()`, `.max()`
- Shape ops: `.reshape()`, `.view()`, `.transpose()`, `.squeeze()`, `.unsqueeze()`
- Activations: `.relu()`, `.sigmoid()`, `.tanh()`
- Math: `.exp()`, `.log()`, `.sqrt()`

#### Neural Network Layers

- `nn.Linear(in_features, out_features)`
- `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`
- `nn.MaxPool2d(kernel_size, stride, padding)`
- `nn.AvgPool2d(kernel_size, stride, padding)`
- `nn.Dropout(p)`
- `nn.BatchNorm1d(num_features)`
- `nn.BatchNorm2d(num_features)`
- `nn.Embedding(num_embeddings, embedding_dim)`

#### Activation Functions

- `nn.ReLU()`
- `nn.Sigmoid()`
- `nn.Tanh()`
- `nn.Softmax(dim)`
- `nn.LogSoftmax(dim)`

#### Loss Functions

- `nn.MSELoss()`
- `nn.CrossEntropyLoss()`
- `nn.BCELoss()`
- `nn.L1Loss()`

#### Optimizers

- `mt.optim.SGD(params, lr, momentum, weight_decay)`
- `mt.optim.Adam(params, lr, betas, eps, weight_decay)`
- `mt.optim.RMSprop(params, lr, alpha, eps, weight_decay)`
- `mt.optim.Adagrad(params, lr, eps, weight_decay)`

### Examples

See the `examples/` directory for complete examples:

- `basic_ops.py` - Basic tensor operations
- `simple_nn.py` - Binary classification with MLP
- `regression.py` - Regression with neural network
- `cnn.py` - Convolutional neural network

### Architecture

```
minitorch/
├── __init__.py          - Package exports
├── tensor.py            - Tensor class and creation functions
├── autograd.py          - Automatic differentiation engine
├── ops.py               - Forward and backward operations
├── functional.py        - Functional API
├── optim.py             - Optimizers
├── cuda_ops.py          - CUDA operation wrappers
├── nn/
│   ├── __init__.py
│   ├── module.py        - Base Module and Parameter classes
│   └── layers.py        - Neural network layers
└── cuda/                - CUDA kernels
    ├── matmul.cu
    ├── elemwise.cu
    ├── reduce.cu
    └── conv.cu
```

### Implementation Details

#### Autograd Engine

MiniTorch implements reverse-mode automatic differentiation using a computational graph. Each operation creates nodes in the graph with saved context for backward pass.

#### CUDA Kernels

Custom CUDA kernels are provided for:
- Matrix multiplication (with shared memory tiling)
- Element-wise operations (add, mul, exp, log, etc.)
- Reduction operations (sum, max, softmax)
- Convolution operations

#### Memory Management

Tensors can exist on CPU (NumPy) or GPU (CuPy). The library handles data movement between devices transparently.

### License

Apache v2.0 License

### Contributing

This is an educational project demonstrating how PyTorch works internally. Feel free to extend it with additional features!
