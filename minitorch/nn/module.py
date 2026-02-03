from ..tensor import Tensor
import numpy as np

class Parameter(Tensor):
  
  def __init__(self, data, requires_grad=True):
    super().__init__(data, requires_grad=requires_grad)
  
  def __repr__(self):
    return f"Parameter({self.data})"

class Module:
  
  def __init__(self):
    self._parameters = {}
    self._modules = {}
    self.training = True
  
  def forward(self, *args, **kwargs):
    raise NotImplementedError
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
  
  def parameters(self):
    params = []
    for param in self._parameters.values():
      params.append(param)
    for module in self._modules.values():
      params.extend(module.parameters())
    return params
  
  def named_parameters(self):
    params = []
    for name, param in self._parameters.items():
      params.append((name, param))
    for name, module in self._modules.items():
      for subname, param in module.named_parameters():
        params.append((f"{name}.{subname}", param))
    return params
  
  def zero_grad(self):
    for param in self.parameters():
      param.zero_grad()
  
  def train(self):
    self.training = True
    for module in self._modules.values():
      module.train()
  
  def eval(self):
    self.training = False
    for module in self._modules.values():
      module.eval()
  
  def to(self, device):
    for param in self.parameters():
      param.data = param.to(device).data
      param.device = device
      if param.grad is not None:
        param.grad.data = param.grad.to(device).data
        param.grad.device = device
    return self
  
  def cpu(self):
    return self.to('cpu')
  
  def cuda(self):
    return self.to('cuda')
  
  def __setattr__(self, name, value):
    if isinstance(value, Parameter):
      self._parameters[name] = value
    elif isinstance(value, Module):
      self._modules[name] = value
    else:
      object.__setattr__(self, name, value)
  
  def __getattr__(self, name):
    if '_parameters' in self.__dict__:
      if name in self._parameters:
        return self._parameters[name]
    if '_modules' in self.__dict__:
      if name in self._modules:
        return self._modules[name]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
