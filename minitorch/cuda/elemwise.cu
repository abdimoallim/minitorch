__global__ void add_kernel(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void sub_kernel(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] - b[idx];
  }
}

__global__ void mul_kernel(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

__global__ void div_kernel(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] / b[idx];
  }
}

__global__ void exp_kernel(float* a, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = expf(a[idx]);
  }
}

__global__ void log_kernel(float* a, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = logf(a[idx] + 1e-8f);
  }
}

__global__ void sqrt_kernel(float* a, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = sqrtf(a[idx]);
  }
}

__global__ void tanh_kernel(float* a, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = tanhf(a[idx]);
  }
}

__global__ void sigmoid_kernel(float* a, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = 1.0f / (1.0f + expf(-a[idx]));
  }
}

__global__ void relu_kernel(float* a, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = fmaxf(0.0f, a[idx]);
  }
}

__global__ void relu_backward_kernel(float* grad_output, float* input, float* grad_input, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    grad_input[idx] = grad_output[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
  }
}

__global__ void pow_kernel(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = powf(a[idx], b[idx]);
  }
}
