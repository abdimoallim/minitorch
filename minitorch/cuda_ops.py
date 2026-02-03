import os
import cupy as cp

_cuda_kernels = {}

def load_kernel(kernel_name, kernel_file, kernel_func):
  if kernel_name in _cuda_kernels:
    return _cuda_kernels[kernel_name]
  
  kernel_dir = os.path.join(os.path.dirname(__file__), 'cuda')
  kernel_path = os.path.join(kernel_dir, kernel_file)
  
  with open(kernel_path, 'r') as f:
    kernel_code = f.read()
  
  module = cp.RawModule(code=kernel_code)
  kernel = module.get_function(kernel_func)
  _cuda_kernels[kernel_name] = kernel
  
  return kernel

def cuda_matmul(a, b):
  M, K = a.shape
  K2, N = b.shape
  assert K == K2, "Dimension mismatch"
  
  c = cp.zeros((M, N), dtype=cp.float32)
  
  block_size = (32, 32)
  grid_size = ((N + block_size[0] - 1) // block_size[0],
               (M + block_size[1] - 1) // block_size[1])
  
  kernel = load_kernel('matmul_shared', 'matmul.cu', 'matmul_shared_kernel')
  kernel(grid_size, block_size, (a, b, c, M, N, K))
  
  return c

def cuda_add(a, b):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('add', 'elemwise.cu', 'add_kernel')
  kernel((grid_size,), (block_size,), (a, b, c, n))
  
  return c

def cuda_sub(a, b):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('sub', 'elemwise.cu', 'sub_kernel')
  kernel((grid_size,), (block_size,), (a, b, c, n))
  
  return c

def cuda_mul(a, b):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('mul', 'elemwise.cu', 'mul_kernel')
  kernel((grid_size,), (block_size,), (a, b, c, n))
  
  return c

def cuda_div(a, b):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('div', 'elemwise.cu', 'div_kernel')
  kernel((grid_size,), (block_size,), (a, b, c, n))
  
  return c

def cuda_exp(a):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('exp', 'elemwise.cu', 'exp_kernel')
  kernel((grid_size,), (block_size,), (a, c, n))
  
  return c

def cuda_log(a):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('log', 'elemwise.cu', 'log_kernel')
  kernel((grid_size,), (block_size,), (a, c, n))
  
  return c

def cuda_relu(a):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('relu', 'elemwise.cu', 'relu_kernel')
  kernel((grid_size,), (block_size,), (a, c, n))
  
  return c

def cuda_sigmoid(a):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('sigmoid', 'elemwise.cu', 'sigmoid_kernel')
  kernel((grid_size,), (block_size,), (a, c, n))
  
  return c

def cuda_tanh(a):
  n = a.size
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = (n + block_size - 1) // block_size
  
  kernel = load_kernel('tanh', 'elemwise.cu', 'tanh_kernel')
  kernel((grid_size,), (block_size,), (a, c, n))
  
  return c

def cuda_softmax(a, dim=-1):
  if dim == -1:
    dim = len(a.shape) - 1
  
  batch_size = a.shape[0] if dim == 1 else 1
  feature_dim = a.shape[dim]
  
  c = cp.zeros_like(a)
  
  block_size = 256
  grid_size = batch_size
  shared_mem_size = 2 * 4
  
  kernel = load_kernel('softmax', 'reduce.cu', 'softmax_kernel')
  kernel((grid_size,), (block_size,), (a, c, batch_size, feature_dim), shared_mem=shared_mem_size)
  
  return c
