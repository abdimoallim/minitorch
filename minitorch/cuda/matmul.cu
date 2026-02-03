__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void matmul_shared_kernel(float* A, float* B, float* C, int M, int N, int K) {
  __shared__ float tileA[32][32];
  __shared__ float tileB[32][32];
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int row = by * 32 + ty;
  int col = bx * 32 + tx;
  
  float sum = 0.0f;
  
  for (int t = 0; t < (K + 31) / 32; t++) {
    if (row < M && t * 32 + tx < K)
      tileA[ty][tx] = A[row * K + t * 32 + tx];
    else
      tileA[ty][tx] = 0.0f;
    
    if (t * 32 + ty < K && col < N)
      tileB[ty][tx] = B[(t * 32 + ty) * N + col];
    else
      tileB[ty][tx] = 0.0f;
    
    __syncthreads();
    
    for (int i = 0; i < 32; i++) {
      sum += tileA[ty][i] * tileB[i][tx];
    }
    
    __syncthreads();
  }
  
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
