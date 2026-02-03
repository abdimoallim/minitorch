import numpy as np

class Optimizer:
  
  def __init__(self, params):
    self.params = list(params)
  
  def step(self):
    raise NotImplementedError
  
  def zero_grad(self):
    for param in self.params:
      param.zero_grad()

class SGD(Optimizer):
  
  def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
    super().__init__(params)
    self.lr = lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.velocities = [np.zeros_like(p.data) for p in self.params]
  
  def step(self):
    for i, param in enumerate(self.params):
      if param.grad is None:
        continue
      
      grad = param.grad.data
      
      if self.weight_decay != 0:
        grad = grad + self.weight_decay * param.data
      
      if self.momentum != 0:
        self.velocities[i] = self.momentum * self.velocities[i] + grad
        grad = self.velocities[i]
      
      param.data = param.data - self.lr * grad

class Adam(Optimizer):
  
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    super().__init__(params)
    self.lr = lr
    self.beta1, self.beta2 = betas
    self.eps = eps
    self.weight_decay = weight_decay
    self.t = 0
    
    self.m = [np.zeros_like(p.data) for p in self.params]
    self.v = [np.zeros_like(p.data) for p in self.params]
  
  def step(self):
    self.t += 1
    
    for i, param in enumerate(self.params):
      if param.grad is None:
        continue
      
      grad = param.grad.data
      
      if self.weight_decay != 0:
        grad = grad + self.weight_decay * param.data
      
      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
      
      m_hat = self.m[i] / (1 - self.beta1 ** self.t)
      v_hat = self.v[i] / (1 - self.beta2 ** self.t)
      
      param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSprop(Optimizer):
  
  def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0):
    super().__init__(params)
    self.lr = lr
    self.alpha = alpha
    self.eps = eps
    self.weight_decay = weight_decay
    
    self.v = [np.zeros_like(p.data) for p in self.params]
  
  def step(self):
    for i, param in enumerate(self.params):
      if param.grad is None:
        continue
      
      grad = param.grad.data
      
      if self.weight_decay != 0:
        grad = grad + self.weight_decay * param.data
      
      self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
      
      param.data = param.data - self.lr * grad / (np.sqrt(self.v[i]) + self.eps)

class Adagrad(Optimizer):
  
  def __init__(self, params, lr=0.01, eps=1e-8, weight_decay=0):
    super().__init__(params)
    self.lr = lr
    self.eps = eps
    self.weight_decay = weight_decay
    
    self.v = [np.zeros_like(p.data) for p in self.params]
  
  def step(self):
    for i, param in enumerate(self.params):
      if param.grad is None:
        continue
      
      grad = param.grad.data
      
      if self.weight_decay != 0:
        grad = grad + self.weight_decay * param.data
      
      self.v[i] += grad ** 2
      
      param.data = param.data - self.lr * grad / (np.sqrt(self.v[i]) + self.eps)
