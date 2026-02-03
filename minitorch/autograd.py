import numpy as np

class Function:
  
  @staticmethod
  def forward(ctx, *args, **kwargs):
    raise NotImplementedError
  
  @staticmethod
  def backward(ctx, grad_output):
    raise NotImplementedError
  
  @classmethod
  def apply(cls, *args, **kwargs):
    ctx = Context()
    output = cls.forward(ctx, *args, **kwargs)
    
    from .tensor import Tensor
    if isinstance(output, Tensor):
      output._ctx = ctx
      output._grad_fn = cls
    
    return output

class Context:
  
  def __init__(self):
    self.saved_tensors = []
    self.saved_values = {}
  
  def save_for_backward(self, *tensors):
    self.saved_tensors = tensors
  
  def save_value(self, name, value):
    self.saved_values[name] = value
