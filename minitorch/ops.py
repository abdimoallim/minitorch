import numpy as np
from .autograd import Function
from .tensor import Tensor

class Add(Function):
  
  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    output = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = grad_output if a.requires_grad else None
    grad_b = grad_output if b.requires_grad else None
    
    if grad_a is not None and a.shape != grad_output.shape:
      grad_a = unbroadcast(grad_output, a.shape)
    if grad_b is not None and b.shape != grad_output.shape:
      grad_b = unbroadcast(grad_output, b.shape)
    
    return grad_a, grad_b

class Sub(Function):
  
  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    output = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = grad_output if a.requires_grad else None
    grad_b = Tensor(-grad_output.data, device=grad_output.device) if b.requires_grad else None
    
    if grad_a is not None and a.shape != grad_output.shape:
      grad_a = unbroadcast(grad_output, a.shape)
    if grad_b is not None and b.shape != grad_output.shape:
      grad_b = unbroadcast(grad_b, b.shape)
    
    return grad_a, grad_b

class Mul(Function):
  
  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    output = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = Tensor(grad_output.data * b.data, device=grad_output.device) if a.requires_grad else None
    grad_b = Tensor(grad_output.data * a.data, device=grad_output.device) if b.requires_grad else None
    
    if grad_a is not None and a.shape != grad_output.shape:
      grad_a = unbroadcast(grad_a, a.shape)
    if grad_b is not None and b.shape != grad_output.shape:
      grad_b = unbroadcast(grad_b, b.shape)
    
    return grad_a, grad_b

class Div(Function):
  
  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    output = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = Tensor(grad_output.data / b.data, device=grad_output.device) if a.requires_grad else None
    grad_b = Tensor(-grad_output.data * a.data / (b.data ** 2), device=grad_output.device) if b.requires_grad else None
    
    if grad_a is not None and a.shape != grad_output.shape:
      grad_a = unbroadcast(grad_a, a.shape)
    if grad_b is not None and b.shape != grad_output.shape:
      grad_b = unbroadcast(grad_b, b.shape)
    
    return grad_a, grad_b

class Pow(Function):
  
  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    output = Tensor(a.data ** b.data, requires_grad=a.requires_grad or b.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = Tensor(grad_output.data * b.data * (a.data ** (b.data - 1)), device=grad_output.device) if a.requires_grad else None
    grad_b = Tensor(grad_output.data * (a.data ** b.data) * np.log(a.data + 1e-8), device=grad_output.device) if b.requires_grad else None
    
    if grad_a is not None and a.shape != grad_output.shape:
      grad_a = unbroadcast(grad_a, a.shape)
    if grad_b is not None and b.shape != grad_output.shape:
      grad_b = unbroadcast(grad_b, b.shape)
    
    return grad_a, grad_b

class Neg(Function):
  
  @staticmethod
  def forward(ctx, a):
    ctx.save_for_backward(a)
    output = Tensor(-a.data, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    grad_a = Tensor(-grad_output.data, device=grad_output.device) if a.requires_grad else None
    return (grad_a,)

class MatMul(Function):
  
  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    output = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = Tensor(grad_output.data @ b.data.T, device=grad_output.device) if a.requires_grad else None
    grad_b = Tensor(a.data.T @ grad_output.data, device=grad_output.device) if b.requires_grad else None
    return grad_a, grad_b

class Sum(Function):
  
  @staticmethod
  def forward(ctx, a, dim, keepdim):
    ctx.save_for_backward(a)
    ctx.save_value('dim', dim)
    ctx.save_value('keepdim', keepdim)
    ctx.save_value('input_shape', a.shape)
    
    result = np.sum(a.data, axis=dim, keepdims=keepdim)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    dim = ctx.saved_values['dim']
    keepdim = ctx.saved_values['keepdim']
    input_shape = ctx.saved_values['input_shape']
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data
    if not keepdim and dim is not None:
      grad = np.expand_dims(grad, axis=dim)
    
    grad = np.broadcast_to(grad, input_shape)
    return (Tensor(grad, device=grad_output.device),)

class Mean(Function):
  
  @staticmethod
  def forward(ctx, a, dim, keepdim):
    ctx.save_for_backward(a)
    ctx.save_value('dim', dim)
    ctx.save_value('keepdim', keepdim)
    ctx.save_value('input_shape', a.shape)
    
    result = np.mean(a.data, axis=dim, keepdims=keepdim)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    dim = ctx.saved_values['dim']
    keepdim = ctx.saved_values['keepdim']
    input_shape = ctx.saved_values['input_shape']
    
    if not a.requires_grad:
      return (None,)
    
    if dim is None:
      n = np.prod(input_shape)
    else:
      n = input_shape[dim]
    
    grad = grad_output.data / n
    if not keepdim and dim is not None:
      grad = np.expand_dims(grad, axis=dim)
    
    grad = np.broadcast_to(grad, input_shape)
    return (Tensor(grad, device=grad_output.device),)

class Max(Function):
  
  @staticmethod
  def forward(ctx, a, dim, keepdim):
    ctx.save_for_backward(a)
    ctx.save_value('dim', dim)
    ctx.save_value('keepdim', keepdim)
    
    if dim is None:
      result = np.max(a.data)
      ctx.save_value('max_indices', np.argmax(a.data))
    else:
      result = np.max(a.data, axis=dim, keepdims=keepdim)
      ctx.save_value('max_indices', np.argmax(a.data, axis=dim))
    
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    dim = ctx.saved_values['dim']
    keepdim = ctx.saved_values['keepdim']
    max_indices = ctx.saved_values['max_indices']
    
    if not a.requires_grad:
      return (None,)
    
    grad = np.zeros_like(a.data)
    
    if dim is None:
      flat_grad = grad.reshape(-1)
      flat_grad[max_indices] = grad_output.data
      grad = flat_grad.reshape(a.shape)
    else:
      if not keepdim:
        grad_output_data = np.expand_dims(grad_output.data, axis=dim)
      else:
        grad_output_data = grad_output.data
      
      np.put_along_axis(grad, np.expand_dims(max_indices, axis=dim), grad_output_data, axis=dim)
    
    return (Tensor(grad, device=grad_output.device),)

class Reshape(Function):
  
  @staticmethod
  def forward(ctx, a, shape):
    ctx.save_for_backward(a)
    ctx.save_value('input_shape', a.shape)
    output = Tensor(a.data.reshape(shape), requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    input_shape = ctx.saved_values['input_shape']
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data.reshape(input_shape)
    return (Tensor(grad, device=grad_output.device),)

class Transpose(Function):
  
  @staticmethod
  def forward(ctx, a, dim0, dim1):
    ctx.save_for_backward(a)
    ctx.save_value('dim0', dim0)
    ctx.save_value('dim1', dim1)
    
    if dim0 is None and dim1 is None:
      if a.ndim == 2:
        result = a.data.T
      else:
        axes = list(range(a.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        result = np.transpose(a.data, axes)
    else:
      axes = list(range(a.ndim))
      axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
      result = np.transpose(a.data, axes)
    
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    dim0 = ctx.saved_values['dim0']
    dim1 = ctx.saved_values['dim1']
    
    if not a.requires_grad:
      return (None,)
    
    if dim0 is None and dim1 is None:
      if a.ndim == 2:
        grad = grad_output.data.T
      else:
        axes = list(range(a.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        grad = np.transpose(grad_output.data, axes)
    else:
      axes = list(range(a.ndim))
      axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
      grad = np.transpose(grad_output.data, axes)
    
    return (Tensor(grad, device=grad_output.device),)

class Unsqueeze(Function):
  
  @staticmethod
  def forward(ctx, a, dim):
    ctx.save_for_backward(a)
    ctx.save_value('dim', dim)
    output = Tensor(np.expand_dims(a.data, axis=dim), requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    dim = ctx.saved_values['dim']
    
    if not a.requires_grad:
      return (None,)
    
    grad = np.squeeze(grad_output.data, axis=dim)
    return (Tensor(grad, device=grad_output.device),)

class Squeeze(Function):
  
  @staticmethod
  def forward(ctx, a, dim):
    ctx.save_for_backward(a)
    ctx.save_value('input_shape', a.shape)
    
    if dim is None:
      result = np.squeeze(a.data)
    else:
      result = np.squeeze(a.data, axis=dim)
    
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    input_shape = ctx.saved_values['input_shape']
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data.reshape(input_shape)
    return (Tensor(grad, device=grad_output.device),)

class Exp(Function):
  
  @staticmethod
  def forward(ctx, a):
    result = np.exp(a.data)
    ctx.save_value('result', result)
    ctx.save_for_backward(a)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    result = ctx.saved_values['result']
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data * result
    return (Tensor(grad, device=grad_output.device),)

class Log(Function):
  
  @staticmethod
  def forward(ctx, a):
    ctx.save_for_backward(a)
    result = np.log(a.data + 1e-8)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data / (a.data + 1e-8)
    return (Tensor(grad, device=grad_output.device),)

class Sqrt(Function):
  
  @staticmethod
  def forward(ctx, a):
    result = np.sqrt(a.data)
    ctx.save_value('result', result)
    ctx.save_for_backward(a)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    result = ctx.saved_values['result']
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data / (2 * result + 1e-8)
    return (Tensor(grad, device=grad_output.device),)

class Tanh(Function):
  
  @staticmethod
  def forward(ctx, a):
    result = np.tanh(a.data)
    ctx.save_value('result', result)
    ctx.save_for_backward(a)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    result = ctx.saved_values['result']
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data * (1 - result ** 2)
    return (Tensor(grad, device=grad_output.device),)

class Sigmoid(Function):
  
  @staticmethod
  def forward(ctx, a):
    result = 1 / (1 + np.exp(-a.data))
    ctx.save_value('result', result)
    ctx.save_for_backward(a)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    result = ctx.saved_values['result']
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data * result * (1 - result)
    return (Tensor(grad, device=grad_output.device),)

class ReLU(Function):
  
  @staticmethod
  def forward(ctx, a):
    ctx.save_for_backward(a)
    result = np.maximum(0, a.data)
    output = Tensor(result, requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    
    if not a.requires_grad:
      return (None,)
    
    grad = grad_output.data * (a.data > 0)
    return (Tensor(grad, device=grad_output.device),)

class Slice(Function):
  
  @staticmethod
  def forward(ctx, a, idx):
    ctx.save_for_backward(a)
    ctx.save_value('idx', idx)
    ctx.save_value('input_shape', a.shape)
    output = Tensor(a.data[idx], requires_grad=a.requires_grad, device=a.device)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    a, = ctx.saved_tensors
    idx = ctx.saved_values['idx']
    input_shape = ctx.saved_values['input_shape']
    
    if not a.requires_grad:
      return (None,)
    
    grad = np.zeros(input_shape)
    grad[idx] = grad_output.data
    return (Tensor(grad, device=grad_output.device),)

def unbroadcast(grad, shape):
  while len(grad.shape) > len(shape):
    grad = Tensor(grad.data.sum(axis=0), device=grad.device)
  
  for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, shape)):
    if grad_dim != orig_dim:
      grad = Tensor(grad.data.sum(axis=i, keepdims=True), device=grad.device)
  
  return grad
