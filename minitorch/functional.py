import numpy as np
from .tensor import Tensor
from .ops import *

def relu(x):
  return x.relu()

def sigmoid(x):
  return x.sigmoid()

def tanh(x):
  return x.tanh()

def softmax(x, dim=-1):
  exp_x = (x - x.max(dim=dim, keepdim=True)).exp()
  return exp_x / exp_x.sum(dim=dim, keepdim=True)

def log_softmax(x, dim=-1):
  return (x - x.max(dim=dim, keepdim=True)) - (x - x.max(dim=dim, keepdim=True)).exp().sum(dim=dim, keepdim=True).log()

def cross_entropy(pred, target):
  log_probs = log_softmax(pred, dim=-1)
  nll = -log_probs * target
  return nll.sum(dim=-1).mean()

def binary_cross_entropy(pred, target):
  return -(target * pred.log() + (1 - target) * (1 - pred).log()).mean()

def mse_loss(pred, target):
  return ((pred - target) ** 2).mean()

def l1_loss(pred, target):
  diff = pred - target
  return (diff * diff.data / (diff.data + 1e-8)).mean()

def dropout(x, p=0.5, training=True):
  if not training or p == 0:
    return x
  
  mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
  return x * Tensor(mask, device=x.device)

def batch_norm(x, running_mean, running_var, weight=None, bias=None, training=True, momentum=0.1, eps=1e-5):
  if training:
    mean = x.mean(dim=0, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=0, keepdim=True)
    
    if running_mean is not None:
      running_mean.data = (1 - momentum) * running_mean.data + momentum * mean.data
    if running_var is not None:
      running_var.data = (1 - momentum) * running_var.data + momentum * var.data
    
    x_normalized = (x - mean) / (var + eps).sqrt()
  else:
    x_normalized = (x - running_mean) / (running_var + eps).sqrt()
  
  if weight is not None:
    x_normalized = x_normalized * weight
  if bias is not None:
    x_normalized = x_normalized + bias
  
  return x_normalized

def conv2d(x, weight, bias=None, stride=1, padding=0):
  batch, in_channels, h, w = x.shape
  out_channels, _, kh, kw = weight.shape
  
  if isinstance(stride, int):
    stride = (stride, stride)
  if isinstance(padding, int):
    padding = (padding, padding)
  
  x_padded = x.data
  if padding[0] > 0 or padding[1] > 0:
    x_padded = np.pad(x_padded, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
  
  out_h = (h + 2 * padding[0] - kh) // stride[0] + 1
  out_w = (w + 2 * padding[1] - kw) // stride[1] + 1
  
  output = np.zeros((batch, out_channels, out_h, out_w))
  
  for b in range(batch):
    for c_out in range(out_channels):
      for i in range(out_h):
        for j in range(out_w):
          h_start = i * stride[0]
          w_start = j * stride[1]
          patch = x_padded[b, :, h_start:h_start+kh, w_start:w_start+kw]
          output[b, c_out, i, j] = np.sum(patch * weight.data[c_out])
          
          if bias is not None:
            output[b, c_out, i, j] += bias.data[c_out]
  
  return Tensor(output, requires_grad=x.requires_grad or weight.requires_grad, device=x.device)

def max_pool2d(x, kernel_size, stride=None, padding=0):
  if stride is None:
    stride = kernel_size
  
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if isinstance(stride, int):
    stride = (stride, stride)
  if isinstance(padding, int):
    padding = (padding, padding)
  
  batch, channels, h, w = x.shape
  kh, kw = kernel_size
  
  x_padded = x.data
  if padding[0] > 0 or padding[1] > 0:
    x_padded = np.pad(x_padded, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant', constant_values=-np.inf)
  
  out_h = (h + 2 * padding[0] - kh) // stride[0] + 1
  out_w = (w + 2 * padding[1] - kw) // stride[1] + 1
  
  output = np.zeros((batch, channels, out_h, out_w))
  
  for b in range(batch):
    for c in range(channels):
      for i in range(out_h):
        for j in range(out_w):
          h_start = i * stride[0]
          w_start = j * stride[1]
          patch = x_padded[b, c, h_start:h_start+kh, w_start:w_start+kw]
          output[b, c, i, j] = np.max(patch)
  
  return Tensor(output, requires_grad=x.requires_grad, device=x.device)

def avg_pool2d(x, kernel_size, stride=None, padding=0):
  if stride is None:
    stride = kernel_size
  
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if isinstance(stride, int):
    stride = (stride, stride)
  if isinstance(padding, int):
    padding = (padding, padding)
  
  batch, channels, h, w = x.shape
  kh, kw = kernel_size
  
  x_padded = x.data
  if padding[0] > 0 or padding[1] > 0:
    x_padded = np.pad(x_padded, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
  
  out_h = (h + 2 * padding[0] - kh) // stride[0] + 1
  out_w = (w + 2 * padding[1] - kw) // stride[1] + 1
  
  output = np.zeros((batch, channels, out_h, out_w))
  
  for b in range(batch):
    for c in range(channels):
      for i in range(out_h):
        for j in range(out_w):
          h_start = i * stride[0]
          w_start = j * stride[1]
          patch = x_padded[b, c, h_start:h_start+kh, w_start:w_start+kw]
          output[b, c, i, j] = np.mean(patch)
  
  return Tensor(output, requires_grad=x.requires_grad, device=x.device)

def embedding(indices, weight):
  indices_data = indices.data.astype(int)
  output = weight.data[indices_data]
  return Tensor(output, requires_grad=weight.requires_grad, device=weight.device)

def one_hot(indices, num_classes):
  indices_data = indices.data.astype(int)
  output = np.eye(num_classes)[indices_data]
  return Tensor(output, device=indices.device)
