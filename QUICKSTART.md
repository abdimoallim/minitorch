Quick Start Guide
==================

Installation
------------

1. Install dependencies:
   pip install numpy

2. For CUDA support (optional):
   pip install cupy-cuda11x

3. Install minitorch:
   pip install -e .


Running Examples
----------------

Basic Operations:
  python examples/basic_ops.py

Simple Neural Network:
  python examples/simple_nn.py

Regression:
  python examples/regression.py

CNN:
  python examples/cnn.py

Complete Pipeline:
  python examples/complete_pipeline.py


Running Tests
-------------

  python test_minitorch.py


Quick Example
-------------

import minitorch as mt
import minitorch.nn as nn

model = nn.Sequential(
  nn.Linear(10, 20),
  nn.ReLU(),
  nn.Linear(20, 1)
)

optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

X = mt.randn(100, 10)
y = mt.randn(100, 1)

for epoch in range(100):
  pred = model(X)
  loss = loss_fn(pred, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if epoch % 20 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


Project Structure
-----------------

minitorch/
├── __init__.py          Main package exports
├── tensor.py            Tensor class with autograd
├── autograd.py          Autograd engine (Function, Context)
├── ops.py               All operations (forward/backward)
├── functional.py        Functional API (F.relu, F.softmax, etc)
├── optim.py             Optimizers (SGD, Adam, RMSprop, Adagrad)
├── cuda_ops.py          CUDA operation wrappers
├── nn/
│   ├── __init__.py
│   ├── module.py        Module and Parameter base classes
│   └── layers.py        Neural network layers
└── cuda/                CUDA kernels (.cu files)
    ├── matmul.cu
    ├── elemwise.cu
    ├── reduce.cu
    └── conv.cu


Key Features
------------

Tensor Operations:
  - Arithmetic: +, -, *, /, **
  - Matrix: @, .t(), .transpose()
  - Reductions: .sum(), .mean(), .max()
  - Shape: .reshape(), .view(), .squeeze(), .unsqueeze()
  - Activations: .relu(), .sigmoid(), .tanh()

Neural Network Layers:
  - Linear, Conv2d
  - MaxPool2d, AvgPool2d
  - BatchNorm1d, BatchNorm2d
  - Dropout
  - ReLU, Sigmoid, Tanh, Softmax
  - Embedding

Loss Functions:
  - MSELoss
  - CrossEntropyLoss
  - BCELoss
  - L1Loss

Optimizers:
  - SGD (with momentum)
  - Adam
  - RMSprop
  - Adagrad

CUDA Support:
  - GPU acceleration via CuPy
  - Custom CUDA kernels for key operations
  - .cuda() and .cpu() methods for device transfer


Tips
----

1. Always set requires_grad=True for parameters you want to optimize

2. Remember to call optimizer.zero_grad() before backward()

3. Use model.train() and model.eval() to switch between training and evaluation

4. For CUDA, move both model and data to GPU:
   model.cuda()
   X = X.cuda()

5. Use .item() to extract scalar values from tensors

6. Check tensor shapes with .shape property


Troubleshooting
---------------

Issue: "ModuleNotFoundError: No module named 'minitorch'"
Solution: Make sure you're in the project root and run:
  pip install -e .
  OR add to your script:
  import sys
  sys.path.insert(0, '/path/to/minitorch')

Issue: "CUDA not available"
Solution: Install CuPy:
  pip install cupy-cuda11x

Issue: Gradient computation errors
Solution: Ensure requires_grad=True for learnable parameters


More Information
----------------

See README.md for comprehensive documentation
See examples/ for complete working examples
See test_minitorch.py for unit tests
