#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include <stdio.h>
#include <math.h>


Layer* initSoftmax(int batch_size, int dim, float* inputs) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    Softmax* softmax = (Softmax*)malloc(sizeof(Softmax));
    softmax->dim = dim;

    MALLOC(&layer->outputs, batch_size * dim * sizeof(float));

    layer->forward = softmax_forward;
    layer->backward = softmax_backward;
    layer->update = softmax_update;
    layer->weights_size = 0;
    MALLOC(&layer->downstream_grads, batch_size * dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = softmax;
    layer->type = LAYER_SOFTMAX;
    return layer;
}


void softmax_forward(Layer* layer, int batch_size) {
    Softmax* softmax = (Softmax*)layer->layer_data;  
    if (USE_CUDA) {
        cuda_softmax_forward(layer->inputs, layer->outputs, batch_size, softmax->dim);
    } else {
        host_softmax_forward(layer->inputs, layer->outputs, batch_size, softmax->dim);
    }
}


void host_softmax_forward(float* inputs, float* outs, int batch_size, int dim) {
    float* sums = (float*)calloc(batch_size, sizeof(float));
    float* max_vals = (float*)calloc(batch_size, sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++){
        float max_val = -INFINITY;
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            if (max_val < inputs[idx]) max_val = inputs[idx];
        }
        max_vals[i] = max_val;
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            sums[i] += exp(inputs[idx] - max_val);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            outs[idx] = exp(inputs[idx] - max_vals[i]) / (sums[i] + 1e-5);
        }
    }
}

__global__ void cuda_softmax_forward_kernel(float* inputs, float* outs, int batch_size, int dim) {
    extern __shared__ float temp[];

    float* shared = temp;
    float* exps = &temp[blockDim.x];
    float* sums = &temp[2 * blockDim.x];

    int i = blockIdx.x;
    int j = threadIdx.x;
    int idx = i * dim + j;

    shared[j] = (j < dim) ? inputs[idx] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) {
            shared[j] = fmaxf(shared[j], shared[j + s]);
        }
        __syncthreads();
    }

    float val = (j < dim) ? expf(inputs[idx] - shared[0]) : 0.0f;
    exps[j] = val;
    sums[j] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) {
            sums[j] += sums[j + s];
        }
        __syncthreads();
    }

    if (j < dim) {
        outs[idx] = val / (sums[0] + 1e-5);
    }
}

void cuda_softmax_forward(float* inputs, float* outs, int batch_size, int dim) {
    dim3 gridDim(batch_size);
    int next_power_of_2 = dim <= 0 ? 1 : (1 << (int)ceil(log2f(dim)));
    dim3 blockDim(next_power_of_2);
    size_t shared_mem_size = 3 * next_power_of_2 * sizeof(float);
    cuda_softmax_forward_kernel<<<gridDim, blockDim, shared_mem_size>>>(inputs, outs, batch_size, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void softmax_backward(Layer* layer, int batch_size) {
    if (USE_CUDA) {
        cuda_softmax_backward(layer->upstream_grads, layer->downstream_grads, layer->outputs, batch_size, ((Softmax *)layer->layer_data)->dim);
    } else {
        host_softmax_backward(layer->upstream_grads, layer->downstream_grads, layer->outputs, batch_size, ((Softmax *)layer->layer_data)->dim);
    }
}

void host_softmax_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            downstream_grads[i * dim + j] = upstream_grads[i * dim + j];
        }
    }
}

__global__ void cuda_softmax_backward_kernel(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim) {
    int i = blockIdx.x / dim;
    int j = blockIdx.x % dim;
    downstream_grads[i * dim + j] = upstream_grads[i * dim + j];
}

void cuda_softmax_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim) {
    dim3 gridDim(batch_size * dim);
    cuda_softmax_backward_kernel<<<gridDim, 1>>>(upstream_grads, downstream_grads, outputs, batch_size, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}



void softmax_update(Layer* layer, int batch_size) {
    
}