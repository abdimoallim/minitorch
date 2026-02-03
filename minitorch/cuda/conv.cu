__global__ void conv2d_forward_kernel(
  float* input, float* weight, float* bias, float* output,
  int batch_size, int in_channels, int out_channels,
  int input_h, int input_w, int kernel_h, int kernel_w,
  int output_h, int output_w, int stride_h, int stride_w,
  int pad_h, int pad_w
) {
  int b = blockIdx.z;
  int c_out = blockIdx.y;
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (b >= batch_size || c_out >= out_channels || out_idx >= output_h * output_w) {
    return;
  }
  
  int out_h = out_idx / output_w;
  int out_w = out_idx % output_w;
  
  float sum = 0.0f;
  
  for (int c_in = 0; c_in < in_channels; c_in++) {
    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        int in_h = out_h * stride_h + kh - pad_h;
        int in_w = out_w * stride_w + kw - pad_w;
        
        if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
          int input_idx = ((b * in_channels + c_in) * input_h + in_h) * input_w + in_w;
          int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
          sum += input[input_idx] * weight[weight_idx];
        }
      }
    }
  }
  
  if (bias != NULL) {
    sum += bias[c_out];
  }
  
  int output_idx = ((b * out_channels + c_out) * output_h + out_h) * output_w + out_w;
  output[output_idx] = sum;
}

__global__ void max_pool2d_forward_kernel(
  float* input, float* output, int* indices,
  int batch_size, int channels,
  int input_h, int input_w,
  int output_h, int output_w,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int pad_h, int pad_w
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * channels * output_h * output_w;
  
  if (idx >= total) return;
  
  int out_w = idx % output_w;
  int out_h = (idx / output_w) % output_h;
  int c = (idx / (output_w * output_h)) % channels;
  int b = idx / (output_w * output_h * channels);
  
  float max_val = -INFINITY;
  int max_idx = -1;
  
  for (int kh = 0; kh < kernel_h; kh++) {
    for (int kw = 0; kw < kernel_w; kw++) {
      int in_h = out_h * stride_h + kh - pad_h;
      int in_w = out_w * stride_w + kw - pad_w;
      
      if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
        int input_idx = ((b * channels + c) * input_h + in_h) * input_w + in_w;
        if (input[input_idx] > max_val) {
          max_val = input[input_idx];
          max_idx = input_idx;
        }
      }
    }
  }
  
  output[idx] = max_val;
  if (indices != NULL) {
    indices[idx] = max_idx;
  }
}
