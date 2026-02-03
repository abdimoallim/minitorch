import numpy as np
from .module import Module, Parameter
from ..tensor import Tensor
from .. import functional as F

class Linear(Module):
  
  def __init__(self, in_features, out_features, bias=True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    
    k = 1.0 / in_features
    self.weight = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_features, in_features)))
    
    if bias:
      self.bias = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_features,)))
    else:
      self.bias = None
  
  def forward(self, x):
    output = x @ self.weight.t()
    if self.bias is not None:
      output = output + self.bias
    return output

class Conv2d(Module):
  
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    
    k = 1.0 / (in_channels * kernel_size[0] * kernel_size[1])
    self.weight = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), 
                                              (out_channels, in_channels, kernel_size[0], kernel_size[1])))
    
    if bias:
      self.bias = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_channels,)))
    else:
      self.bias = None
  
  def forward(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

class MaxPool2d(Module):
  
  def __init__(self, kernel_size, stride=None, padding=0):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
  
  def forward(self, x):
    return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

class AvgPool2d(Module):
  
  def __init__(self, kernel_size, stride=None, padding=0):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
  
  def forward(self, x):
    return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

class Dropout(Module):
  
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p
  
  def forward(self, x):
    return F.dropout(x, self.p, self.training)

class BatchNorm1d(Module):
  
  def __init__(self, num_features, eps=1e-5, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    
    self.weight = Parameter(np.ones(num_features))
    self.bias = Parameter(np.zeros(num_features))
    
    self.running_mean = np.zeros(num_features)
    self.running_var = np.ones(num_features)
  
  def forward(self, x):
    running_mean_tensor = Tensor(self.running_mean.copy(), requires_grad=False)
    running_var_tensor = Tensor(self.running_var.copy(), requires_grad=False)
    
    result = F.batch_norm(x, running_mean_tensor, running_var_tensor, 
                         self.weight, self.bias, self.training, self.momentum, self.eps)
    
    if self.training:
      self.running_mean = running_mean_tensor.data.copy()
      self.running_var = running_var_tensor.data.copy()
    
    return result

class BatchNorm2d(Module):
  
  def __init__(self, num_features, eps=1e-5, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    
    self.weight = Parameter(np.ones(num_features))
    self.bias = Parameter(np.zeros(num_features))
    
    self.running_mean = np.zeros(num_features)
    self.running_var = np.ones(num_features)
  
  def forward(self, x):
    batch, channels, height, width = x.shape
    x_reshaped = x.transpose(0, 1).reshape(channels, -1).transpose(0, 1)
    
    running_mean_tensor = Tensor(self.running_mean.copy(), requires_grad=False)
    running_var_tensor = Tensor(self.running_var.copy(), requires_grad=False)
    
    out = F.batch_norm(x_reshaped, running_mean_tensor, running_var_tensor,
                      self.weight, self.bias, self.training, self.momentum, self.eps)
    
    if self.training:
      self.running_mean = running_mean_tensor.data.copy()
      self.running_var = running_var_tensor.data.copy()
    
    out = out.transpose(0, 1).reshape(channels, batch, height, width).transpose(0, 1)
    return out

class ReLU(Module):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    return F.relu(x)

class Sigmoid(Module):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    return F.sigmoid(x)

class Tanh(Module):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    return F.tanh(x)

class Softmax(Module):
  
  def __init__(self, dim=-1):
    super().__init__()
    self.dim = dim
  
  def forward(self, x):
    return F.softmax(x, self.dim)

class LogSoftmax(Module):
  
  def __init__(self, dim=-1):
    super().__init__()
    self.dim = dim
  
  def forward(self, x):
    return F.log_softmax(x, self.dim)

class Embedding(Module):
  
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    
    self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.01)
  
  def forward(self, indices):
    return F.embedding(indices, self.weight)

class Sequential(Module):
  
  def __init__(self, *layers):
    super().__init__()
    for idx, layer in enumerate(layers):
      self._modules[str(idx)] = layer
  
  def forward(self, x):
    for layer in self._modules.values():
      x = layer(x)
    return x

class MSELoss(Module):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, pred, target):
    return F.mse_loss(pred, target)

class CrossEntropyLoss(Module):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, pred, target):
    return F.cross_entropy(pred, target)

class BCELoss(Module):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, pred, target):
    return F.binary_cross_entropy(pred, target)

class L1Loss(Module):
  
  def __init__(self):
    super().__init__()
  
  def forward(self, pred, target):
    return F.l1_loss(pred, target)
