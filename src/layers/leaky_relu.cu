#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include <stdio.h>


Layer* initLeakyReLU(int batch_size, int dim, int coeff, float* inputs, int id) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    LeakyReLU* leakyReLU = (LeakyReLU*)malloc(sizeof(LeakyReLU));
    leakyReLU->coeff = coeff;
    leakyReLU->dim = dim;

    MALLOC(&layer->outputs, batch_size * dim * sizeof(float));

    layer->id = id;
    layer->forward = leakyReLU_forward;
    layer->backward = leakyReLU_backward;
    layer->update = leakyReLU_update;
    layer->weights_size = 0;
    MALLOC(&layer->downstream_grads, batch_size * dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = leakyReLU;
    layer->type = LAYER_LEAKY_RELU;
    return layer;
}


void leakyReLU_forward(Layer* layer, int batch_size) { 
    LeakyReLU* leakyReLU = (LeakyReLU*)layer->layer_data;  
    
    if (USE_CUDA) {
        cuda_leakyReLU_forward(layer->inputs, layer->outputs, batch_size, leakyReLU->dim, leakyReLU->coeff);
    } else {
        host_leakyReLU_forward(layer->inputs, layer->outputs, batch_size, leakyReLU->dim, leakyReLU->coeff);
    }
}


void host_leakyReLU_forward(float* inputs, float* outs, int batch_size, int dim, float coeff) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            outs[idx] = (inputs[idx] > 0) ? inputs[idx] : (coeff * inputs[idx]);
        }
    }
}

__global__ void cuda_leakyReLU_forward_kernel(float* inputs, float* outs, int batch_size, int dim, float coeff) {
    int i = blockIdx.x / dim;
    int j = blockIdx.x % dim;
    int idx = i * dim + j;
    outs[idx] = (inputs[idx] > 0) ? inputs[idx] : (coeff * inputs[idx]);
}

void cuda_leakyReLU_forward(float* inputs, float* outs, int batch_size, int dim, float coeff) {
    dim3 gridDim(batch_size * dim);
    cuda_leakyReLU_forward_kernel<<<gridDim, 1>>>(inputs, outs, batch_size, dim, coeff);
    CUDA_CHECK(cudaDeviceSynchronize());
}


void leakyReLU_backward(Layer* layer, int batch_size) {
    LeakyReLU* leakyReLU = (LeakyReLU*)layer->layer_data;
    float* inputs = layer->inputs;
    float* upstream_grads = layer->upstream_grads;
    float* downstream_grads = layer->downstream_grads;
    float coeff = leakyReLU->coeff;
    int dim = leakyReLU->dim;

    if (USE_CUDA) {
        cuda_leakyReLU_backward(inputs, upstream_grads, downstream_grads, coeff, batch_size, dim);
    } else {
        host_leakyReLU_backward(inputs, upstream_grads, downstream_grads, coeff, batch_size, dim);

    }
}

void host_leakyReLU_backward(float* inputs, float* upstream_grads, float* downstream_grads, float coeff, int batch_size, int dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            if (inputs[idx] >= 0) {
                downstream_grads[idx] = upstream_grads[idx];
            } else {
                downstream_grads[idx] = coeff * upstream_grads[idx];
            }
        }
    }
}

__global__ void cuda_leakyReLU_backward_kernel(float* inputs, float* upstream_grads, float* downstream_grads, float coeff, int batch_size, int dim) {
    int i = blockIdx.x / dim;
    int j = blockIdx.x % dim;
    int idx = i * dim + j;

    if (inputs[idx] >= 0) {
        downstream_grads[idx] = upstream_grads[idx];
    } else {
        downstream_grads[idx] = coeff * upstream_grads[idx];
    }
}

void cuda_leakyReLU_backward(float* inputs, float* upstream_grads, float* downstream_grads, float coeff, int batch_size, int dim) {
    dim3 gridDim(batch_size * dim);
    cuda_leakyReLU_backward_kernel<<<gridDim, 1>>>(inputs, upstream_grads, downstream_grads, coeff, batch_size, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}


void leakyReLU_update(Layer* layer, int batch_size) {
    
}