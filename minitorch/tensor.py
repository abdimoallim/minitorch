import numpy as np
from typing import Optional, Union, Tuple, List
from .autograd import Function, Context

class Tensor:
  
  def __init__(self, data, requires_grad=False, device='cpu'):
    if isinstance(data, Tensor):
      self.data = data.data
    elif isinstance(data, np.ndarray):
      self.data = data
    else:
      self.data = np.array(data)
    
    self.requires_grad = requires_grad
    self.grad = None
    self._grad_fn = None
    self._ctx = None
    self.device = device
    
    if device == 'cuda':
      self._to_cuda()
  
  def _to_cuda(self):
    try:
      import cupy as cp
      if isinstance(self.data, np.ndarray):
        self.data = cp.array(self.data)
    except ImportError:
      raise RuntimeError("CuPy not installed. Cannot use CUDA.")
  
  def _to_cpu(self):
    try:
      import cupy as cp
      if isinstance(self.data, cp.ndarray):
        self.data = cp.asnumpy(self.data)
    except ImportError:
      pass
  
  @property
  def shape(self):
    return self.data.shape
  
  @property
  def ndim(self):
    return self.data.ndim
  
  @property
  def dtype(self):
    return self.data.dtype
  
  @property
  def size(self):
    return self.data.size
  
  def numpy(self):
    if self.device == 'cuda':
      import cupy as cp
      return cp.asnumpy(self.data)
    return self.data
  
  def item(self):
    return self.data.item()
  
  def to(self, device):
    if device == self.device:
      return self
    
    new_tensor = Tensor(self.data, requires_grad=self.requires_grad, device='cpu')
    if device == 'cuda':
      new_tensor._to_cuda()
      new_tensor.device = 'cuda'
    return new_tensor
  
  def cpu(self):
    return self.to('cpu')
  
  def cuda(self):
    return self.to('cuda')
  
  def backward(self, grad=None):
    if not self.requires_grad:
      return
    
    if grad is None:
      if self.data.size == 1:
        grad = Tensor(np.ones_like(self.data), device=self.device)
      else:
        raise RuntimeError("grad must be specified for non-scalar tensors")
    
    if self.grad is None:
      self.grad = Tensor(grad.data.copy(), requires_grad=False, device=self.device)
    else:
      self.grad = Tensor(self.grad.data + grad.data, requires_grad=False, device=self.device)
    
    if self._grad_fn is not None:
      grads = self._grad_fn.backward(self._ctx, grad)
      
      if not isinstance(grads, tuple):
        grads = (grads,)
      
      for tensor, grad in zip(self._ctx.saved_tensors, grads):
        if tensor.requires_grad and grad is not None:
          tensor.backward(grad)
  
  def zero_grad(self):
    self.grad = None
  
  def __repr__(self):
    return f"Tensor({self.data}, requires_grad={self.requires_grad})"
  
  def __add__(self, other):
    from .ops import Add
    return Add.apply(self, other if isinstance(other, Tensor) else Tensor(other, device=self.device))
  
  def __radd__(self, other):
    return self.__add__(other)
  
  def __mul__(self, other):
    from .ops import Mul
    return Mul.apply(self, other if isinstance(other, Tensor) else Tensor(other, device=self.device))
  
  def __rmul__(self, other):
    return self.__mul__(other)
  
  def __sub__(self, other):
    from .ops import Sub
    return Sub.apply(self, other if isinstance(other, Tensor) else Tensor(other, device=self.device))
  
  def __rsub__(self, other):
    from .ops import Sub
    return Sub.apply(Tensor(other, device=self.device), self)
  
  def __truediv__(self, other):
    from .ops import Div
    return Div.apply(self, other if isinstance(other, Tensor) else Tensor(other, device=self.device))
  
  def __rtruediv__(self, other):
    from .ops import Div
    return Div.apply(Tensor(other, device=self.device), self)
  
  def __pow__(self, other):
    from .ops import Pow
    return Pow.apply(self, other if isinstance(other, Tensor) else Tensor(other, device=self.device))
  
  def __neg__(self):
    from .ops import Neg
    return Neg.apply(self)
  
  def __matmul__(self, other):
    from .ops import MatMul
    return MatMul.apply(self, other)
  
  def __getitem__(self, idx):
    from .ops import Slice
    return Slice.apply(self, idx)
  
  def sum(self, dim=None, keepdim=False):
    from .ops import Sum
    return Sum.apply(self, dim, keepdim)
  
  def mean(self, dim=None, keepdim=False):
    from .ops import Mean
    return Mean.apply(self, dim, keepdim)
  
  def max(self, dim=None, keepdim=False):
    from .ops import Max
    return Max.apply(self, dim, keepdim)
  
  def reshape(self, *shape):
    from .ops import Reshape
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    return Reshape.apply(self, shape)
  
  def view(self, *shape):
    return self.reshape(*shape)
  
  def transpose(self, dim0=None, dim1=None):
    from .ops import Transpose
    return Transpose.apply(self, dim0, dim1)
  
  def t(self):
    return self.transpose()
  
  def unsqueeze(self, dim):
    from .ops import Unsqueeze
    return Unsqueeze.apply(self, dim)
  
  def squeeze(self, dim=None):
    from .ops import Squeeze
    return Squeeze.apply(self, dim)
  
  def exp(self):
    from .ops import Exp
    return Exp.apply(self)
  
  def log(self):
    from .ops import Log
    return Log.apply(self)
  
  def sqrt(self):
    from .ops import Sqrt
    return Sqrt.apply(self)
  
  def tanh(self):
    from .ops import Tanh
    return Tanh.apply(self)
  
  def sigmoid(self):
    from .ops import Sigmoid
    return Sigmoid.apply(self)
  
  def relu(self):
    from .ops import ReLU
    return ReLU.apply(self)
  
  def contiguous(self):
    return self
  
  def detach(self):
    return Tensor(self.data.copy(), requires_grad=False, device=self.device)

def tensor(data, requires_grad=False, device='cpu'):
  return Tensor(data, requires_grad=requires_grad, device=device)

def zeros(*shape, requires_grad=False, device='cpu'):
  return Tensor(np.zeros(shape), requires_grad=requires_grad, device=device)

def ones(*shape, requires_grad=False, device='cpu'):
  return Tensor(np.ones(shape), requires_grad=requires_grad, device=device)

def randn(*shape, requires_grad=False, device='cpu'):
  return Tensor(np.random.randn(*shape), requires_grad=requires_grad, device=device)

def rand(*shape, requires_grad=False, device='cpu'):
  return Tensor(np.random.rand(*shape), requires_grad=requires_grad, device=device)

def arange(start, end=None, step=1, requires_grad=False, device='cpu'):
  if end is None:
    end = start
    start = 0
  return Tensor(np.arange(start, end, step), requires_grad=requires_grad, device=device)

def eye(n, m=None, requires_grad=False, device='cpu'):
  if m is None:
    m = n
  return Tensor(np.eye(n, m), requires_grad=requires_grad, device=device)

def empty(*shape, requires_grad=False, device='cpu'):
  return Tensor(np.empty(shape), requires_grad=requires_grad, device=device)
