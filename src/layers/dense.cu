#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include <stdio.h>
#include <math.h>
#include <mpi.h>


Layer* initDenseLayer(int batch_size, int in_dim, int out_dim, float* inputs, int id) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    DenseLayer* denseLayer = (DenseLayer*)malloc(sizeof(DenseLayer));
    denseLayer->in_dim = in_dim;
    denseLayer->out_dim = out_dim;

    layer->weights_size = (in_dim + 1) * out_dim * sizeof(float);
    
    MALLOC(&layer->outputs, batch_size * out_dim * sizeof(float));
    MALLOC(&denseLayer->inputs_augmented, batch_size * (in_dim + 1) * sizeof(float));
    MALLOC(&denseLayer->inputs_augmented_T, batch_size * (in_dim + 1) * sizeof(float));
    MALLOC((void**)&layer->weights, layer->weights_size);
    MALLOC((void**)&layer->weights_grads, layer->weights_size);
    MALLOC(&denseLayer->weights_T, layer->weights_size);

    srand(42);
    float scale = sqrtf(2.0f / (in_dim + out_dim)); // He initialization for ReLU-like networks
    for (int i = 0; i < (in_dim + 1) * out_dim; i++) {
        layer->weights[i] = scale * ((float)rand() / RAND_MAX - 0.5f); // Random [-scale/2, scale/2]
    }

    layer->id = id;
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->update = dense_update;
    MALLOC(&layer->downstream_grads, batch_size * in_dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = denseLayer;
    layer->type = LAYER_DENSE;
    return layer;
}

void host_populate_inputs_augmented(float* inputs, float* inputs_augmented, int batch_size, int in_dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_dim; j++) {
            inputs_augmented[i * (in_dim + 1) + j] = inputs[i * in_dim + j];
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        inputs_augmented[i * (in_dim + 1) + in_dim] = 1.;
    }
}

__global__ void cuda_populate_inputs_augmented_kernel(float* inputs, float* inputs_augmented, int batch_size, int in_dim) {
    int i = blockIdx.x / in_dim;
    int j = blockIdx.x % in_dim;
    inputs_augmented[i * (in_dim + 1) + j] = inputs[i * in_dim + j];
    if (j == 0) {
        inputs_augmented[i * (in_dim + 1) + in_dim] = 1.;
    }
}

void cuda_populate_inputs_augmented(float* inputs, float* inputs_augmented, int batch_size, int in_dim) {
    dim3 gridDim(batch_size * in_dim);
    cuda_populate_inputs_augmented_kernel<<<gridDim, 1>>>(inputs, inputs_augmented, batch_size, in_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}


void dense_forward(Layer* layer, int batch_size) {  
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 
    
    float* weights = layer->weights;
    float* inputs = layer->inputs;
    float* inputs_augmented = denseLayer->inputs_augmented;
    float* outs = layer->outputs;
    int in_dim = denseLayer->in_dim;
    int out_dim = denseLayer->out_dim;

    if (USE_CUDA) {
        cuda_populate_inputs_augmented(inputs, inputs_augmented, batch_size, in_dim);
        cuda_matrix_multiply(inputs_augmented, weights, outs, batch_size, in_dim + 1, out_dim);
    } else {
        host_populate_inputs_augmented(inputs, inputs_augmented, batch_size, in_dim);
        host_matrix_multiply(inputs_augmented, weights, outs, batch_size, in_dim + 1, out_dim);
    }
}


void dense_backward(Layer* layer, int batch_size) {

    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 

    int in_dim = denseLayer->in_dim;
    int out_dim = denseLayer->out_dim;
    float* weights = layer->weights;
    float* weights_T = denseLayer->weights_T;
    float* weights_grads = layer->weights_grads;
    float* inputs_augmented = denseLayer->inputs_augmented;
    float* inputs_augmented_T = denseLayer->inputs_augmented_T;

    if (USE_CUDA) {
        cuda_transpose(inputs_augmented, inputs_augmented_T, batch_size, in_dim + 1);
        cuda_matrix_multiply(inputs_augmented_T, layer->upstream_grads, weights_grads, in_dim + 1, batch_size, out_dim);
        cuda_transpose(weights, weights_T, in_dim + 1, out_dim);
        cuda_matrix_multiply(layer->upstream_grads, weights_T, inputs_augmented, batch_size, out_dim, in_dim + 1); // reuse of memory allocated for inputs augmented
        cuda_extract_downstream_grads(layer->downstream_grads, inputs_augmented, batch_size, in_dim);
    } else {
        host_transpose(inputs_augmented, inputs_augmented_T, batch_size, in_dim + 1);
        host_matrix_multiply(inputs_augmented_T, layer->upstream_grads, weights_grads, in_dim + 1, batch_size, out_dim);
        host_transpose(weights, weights_T, in_dim + 1, out_dim);
        host_matrix_multiply(layer->upstream_grads, weights_T, inputs_augmented, batch_size, out_dim, in_dim + 1); // reuse of memory allocated for inputs augmented
        host_extract_downstream_grads(layer->downstream_grads, inputs_augmented, batch_size, in_dim);
    }

    if (!USE_MPI_WEIGHT_SHARING) {
        dense_update(layer, batch_size);
    }
}


void dense_update(Layer* layer, int batch_size) {
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 
    float* weights = layer->weights;
    float* weights_grads = layer->weights_grads;
    int in_dim = denseLayer->in_dim;
    int out_dim = denseLayer->out_dim;

    if (USE_CUDA) {
        cuda_weight_update(weights, weights_grads, batch_size, in_dim, out_dim);
    } else {
        host_weight_update(weights, weights_grads, batch_size, in_dim, out_dim);
    }
}


void host_extract_downstream_grads(float* downstream_grads, float* inputs_augmented, int batch_size, int in_dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_dim; j++) {
            downstream_grads[i * in_dim + j] = inputs_augmented[i * (in_dim + 1) + j];
        }
    }
}

__global__ void cuda_extract_downstream_grads_kernel(float* downstream_grads, float* inputs_augmented, int batch_size, int in_dim) {
    int i = blockIdx.x / in_dim;
    int j = blockIdx.x % in_dim;
    downstream_grads[i * in_dim + j] = inputs_augmented[i * (in_dim + 1) + j];
}

void cuda_extract_downstream_grads(float* downstream_grads, float* inputs_augmented, int batch_size, int in_dim) {
    dim3 gridDim(batch_size * in_dim);
    cuda_extract_downstream_grads_kernel<<<gridDim, 1>>>(downstream_grads, inputs_augmented, batch_size, in_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}


void host_weight_update(float* weights, float* weights_grads, int batch_size, int in_dim, int out_dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < in_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            weights[i * out_dim + j] -= LR * weights_grads[i * out_dim + j] / batch_size;
        }
    }
}

__global__ void cuda_weight_update_kernel(float* weights, float* weights_grads, int batch_size, int in_dim, int out_dim) {
    int i = blockIdx.x / out_dim;
    int j = blockIdx.x % out_dim;
    weights[i * out_dim + j] -= LR * weights_grads[i * out_dim + j] / batch_size;
}

void cuda_weight_update(float* weights, float* weights_grads, int batch_size, int in_dim, int out_dim) {
    dim3 gridDim(in_dim * out_dim);
    cuda_weight_update_kernel<<<gridDim, 1>>>(weights, weights_grads, batch_size, in_dim, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}
