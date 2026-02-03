__global__ void reduce_sum_kernel(float* input, float* output, int n, int stride) {
  extern __shared__ float sdata[];
  
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  sdata[tid] = (idx < n) ? input[idx * stride] : 0.0f;
  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

__global__ void reduce_max_kernel(float* input, float* output, int n, int stride) {
  extern __shared__ float sdata[];
  
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  sdata[tid] = (idx < n) ? input[idx * stride] : -INFINITY;
  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    atomicMax((int*)output, __float_as_int(sdata[0]));
  }
}

__global__ void softmax_kernel(float* input, float* output, int batch_size, int dim) {
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x;
  
  extern __shared__ float shared_mem[];
  float* max_val = shared_mem;
  float* sum_val = &shared_mem[1];
  
  if (batch_idx >= batch_size) return;
  
  float* input_row = input + batch_idx * dim;
  float* output_row = output + batch_idx * dim;
  
  float local_max = -INFINITY;
  for (int i = tid; i < dim; i += blockDim.x) {
    local_max = fmaxf(local_max, input_row[i]);
  }
  
  __shared__ float temp[256];
  temp[tid] = local_max;
  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      temp[tid] = fmaxf(temp[tid], temp[tid + s]);
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    max_val[0] = temp[0];
  }
  __syncthreads();
  
  float local_sum = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    local_sum += expf(input_row[i] - max_val[0]);
  }
  
  temp[tid] = local_sum;
  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      temp[tid] += temp[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    sum_val[0] = temp[0];
  }
  __syncthreads();
  
  for (int i = tid; i < dim; i += blockDim.x) {
    output_row[i] = expf(input_row[i] - max_val[0]) / sum_val[0];
  }
}
