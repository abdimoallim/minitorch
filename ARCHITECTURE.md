MiniTorch Architecture
======================

Overview
--------

MiniTorch is structured in layers, from low-level tensor operations to high-level neural network modules.


Layer 1: Core Tensor & Autograd
--------------------------------

Tensor (tensor.py)
  - Core data structure holding NumPy/CuPy arrays
  - Tracks gradients and computational graph
  - Properties: data, grad, requires_grad, device, shape
  - Methods: backward(), zero_grad(), to(), cuda(), cpu()

Function & Context (autograd.py)
  - Base class for all differentiable operations
  - Context saves information needed for backward pass
  - Each operation implements forward() and backward()

Example Flow:
  z = x + y
  1. Add.forward(x, y) -> computes z = x + y
  2. Saves x, y in context
  3. z._grad_fn = Add, z._ctx = context
  4. z.backward() -> calls Add.backward() -> computes dx, dy


Layer 2: Operations
--------------------

Operations (ops.py)
  All differentiable operations inherit from Function:
  
  Arithmetic:
    - Add, Sub, Mul, Div, Pow, Neg
    - Handle broadcasting automatically
  
  Matrix:
    - MatMul: implements @ operator
    - Uses numpy/cupy dot product
  
  Reductions:
    - Sum, Mean, Max
    - Support axis-wise operations
  
  Shape:
    - Reshape, Transpose, Squeeze, Unsqueeze
    - Slice for indexing
  
  Activations:
    - ReLU, Sigmoid, Tanh
    - Exp, Log, Sqrt
  
  Each operation:
    1. forward(): compute result
    2. backward(): compute gradient w.r.t inputs
    3. Handles gradient accumulation


Layer 3: Functional API
------------------------

Functional (functional.py)
  High-level functions built on operations:
  
  - relu(x), sigmoid(x), tanh(x)
  - softmax(x, dim), log_softmax(x, dim)
  - cross_entropy(pred, target)
  - binary_cross_entropy(pred, target)
  - mse_loss(pred, target)
  - dropout(x, p, training)
  - batch_norm(x, mean, var, weight, bias, ...)
  - conv2d(x, weight, bias, stride, padding)
  - max_pool2d(x, kernel_size, stride, padding)
  - avg_pool2d(x, kernel_size, stride, padding)


Layer 4: Neural Network Modules
--------------------------------

Module (nn/module.py)
  Base class for all neural network layers:
  
  - Manages parameters (weights, biases)
  - Manages sub-modules (for nested models)
  - Training vs evaluation mode
  - Device management (CPU/CUDA)
  
  Key methods:
    - forward(*args): defines computation
    - parameters(): returns all learnable params
    - zero_grad(): clears gradients
    - train()/eval(): switch modes
    - to(device): move to CPU/CUDA

Parameter (nn/module.py)
  Special Tensor subclass for learnable parameters
  - Always has requires_grad=True
  - Automatically registered in Module

Layers (nn/layers.py)
  
  Linear Layers:
    - Linear(in, out, bias=True)
      Weight init: U(-√k, √k) where k=1/in_features
  
  Convolutional:
    - Conv2d(in_ch, out_ch, kernel, stride, padding)
      Implements 2D convolution (naive Python)
  
  Pooling:
    - MaxPool2d(kernel, stride, padding)
    - AvgPool2d(kernel, stride, padding)
  
  Normalization:
    - BatchNorm1d(features)
    - BatchNorm2d(features)
      Tracks running mean/var
  
  Regularization:
    - Dropout(p=0.5)
      Only active in training mode
  
  Activations:
    - ReLU(), Sigmoid(), Tanh()
    - Softmax(dim), LogSoftmax(dim)
  
  Other:
    - Embedding(num_embeddings, dim)
    - Sequential(*layers)
  
  Loss Functions:
    - MSELoss(), L1Loss()
    - CrossEntropyLoss()
    - BCELoss()


Layer 5: Optimizers
--------------------

Optimizer (optim.py)
  Base class managing parameter updates
  
  SGD:
    - Supports momentum
    - Supports weight decay
    Update: θ = θ - lr * (grad + wd * θ)
  
  Adam:
    - Adaptive learning rates
    - First and second moment estimates
    - Bias correction
  
  RMSprop:
    - Moving average of squared gradients
    - Adaptive learning rates
  
  Adagrad:
    - Accumulates squared gradients
    - Learning rate adapts per parameter


Layer 6: CUDA Acceleration
---------------------------

CUDA Kernels (cuda/*.cu)
  
  matmul.cu:
    - matmul_kernel: naive implementation
    - matmul_shared_kernel: uses shared memory
  
  elemwise.cu:
    - add, sub, mul, div
    - exp, log, sqrt, tanh, sigmoid, relu
    - pow, relu_backward
  
  reduce.cu:
    - reduce_sum_kernel: parallel reduction
    - reduce_max_kernel: parallel max
    - softmax_kernel: numerically stable softmax
  
  conv.cu:
    - conv2d_forward_kernel: 2D convolution
    - max_pool2d_forward_kernel: max pooling

CUDA Ops (cuda_ops.py)
  Python wrappers for CUDA kernels using CuPy
  - Load and compile kernels
  - Launch with appropriate grid/block sizes
  - Handle memory management


Data Flow Example
-----------------

Training a simple model:

1. User Code:
   model = nn.Linear(10, 1)
   optimizer = Adam(model.parameters(), lr=0.01)
   
2. Forward Pass:
   pred = model(x)
   
   Flow:
   x -> Linear.forward() -> matmul(x, W.t()) + b
                         -> creates computational graph
   
3. Loss Computation:
   loss = MSELoss()(pred, y)
   
   Flow:
   (pred - y)**2 -> mean() -> scalar loss
                           -> graph continues
   
4. Backward Pass:
   loss.backward()
   
   Flow:
   loss.backward(grad=1.0)
   -> calls Mean.backward()
   -> calls Pow.backward()
   -> calls Sub.backward()
   -> calls Add.backward()
   -> calls MatMul.backward()
   -> accumulates W.grad and b.grad
   
5. Optimizer Step:
   optimizer.step()
   
   Flow:
   For each parameter p:
     update m, v (Adam statistics)
     p.data = p.data - lr * m / sqrt(v)


Memory Layout
-------------

Tensor Storage:
  CPU: NumPy array (row-major, C-contiguous)
  GPU: CuPy array (mirrors NumPy on GPU)

Gradient Storage:
  Separate tensor stored in .grad attribute
  Accumulated during backward pass
  Cleared by zero_grad()

Parameter Storage:
  Stored in Module._parameters dict
  Nested modules in Module._modules dict
  Recursive traversal for parameters()


Design Principles
-----------------

1. PyTorch-like API
   - Familiar interface for PyTorch users
   - Same method names and behaviors
   
2. Educational Focus
   - Clear, readable code
   - Explicit implementations
   - No hidden magic
   
3. Modular Design
   - Separation of concerns
   - Composable components
   - Easy to extend
   
4. Gradient Computation
   - Automatic differentiation
   - Dynamic computation graphs
   - Reverse-mode autodiff


Performance Considerations
--------------------------

CPU:
  - NumPy BLAS for matmul
  - Vectorized operations
  - Python loops for conv/pool (slow but clear)

GPU:
  - CuPy for GPU arrays
  - Custom CUDA kernels for key ops
  - Shared memory in matmul kernel
  - Parallel reduction patterns

Bottlenecks:
  - Conv2d/Pooling: naive Python implementation
  - Gradient accumulation: creates new arrays
  - Graph construction: Python overhead


Extension Points
----------------

1. New Operations:
   - Subclass Function
   - Implement forward() and backward()
   - Add method to Tensor class

2. New Layers:
   - Subclass Module
   - Implement __init__() and forward()
   - Register parameters with self.param = Parameter(...)

3. New Optimizers:
   - Subclass Optimizer
   - Implement step()
   - Maintain optimizer state in __init__

4. CUDA Kernels:
   - Write .cu file in cuda/
   - Add wrapper in cuda_ops.py
   - Call from operation's forward()


Limitations
-----------

1. No in-place operations
2. No sparse tensors
3. Limited broadcasting rules
4. No gradient checkpointing
5. No mixed precision training
6. Simple parameter initialization
7. No model serialization
8. Naive convolution implementation


Future Improvements
-------------------

1. Im2col for faster convolution
2. Better memory management
3. In-place operations
4. JIT compilation
5. Model save/load
6. More optimizers (AdamW, etc)
7. Learning rate schedulers
8. Data loading utilities
9. Distributed training
10. More CUDA kernels
