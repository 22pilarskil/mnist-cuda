#include "../../include/model.h"
#include "../../include/utils.h"
#include "../../include/layers/dense.h"
#include <stdio.h>
#include <math.h>
#include <mpi.h>



Layer* initDenseLayer(int batch_size, int in_dim, int out_dim, float* inputs, int id) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    DenseLayer* denseLayer = (DenseLayer*)malloc(sizeof(DenseLayer));
    denseLayer->in_dim = in_dim;
    denseLayer->out_dim = out_dim;
    denseLayer->id = id;

    layer->weights_size = (in_dim + 1) * out_dim * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(&layer->outputs, batch_size * out_dim * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&denseLayer->inputs_augmented, batch_size * (in_dim + 1) * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&denseLayer->inputs_augmented_T, batch_size * (in_dim + 1) * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged((void**)&layer->weights, layer->weights_size));
    CUDA_CHECK(cudaMallocManaged((void**)&layer->weights_grads, layer->weights_size));
    CUDA_CHECK(cudaMallocManaged(&denseLayer->weights_T, layer->weights_size));

    srand(42);
    float scale = sqrtf(2.0f / (in_dim + out_dim)); // He initialization for ReLU-like networks
    for (int i = 0; i < (in_dim + 1) * out_dim; i++) {
        layer->weights[i] = scale * ((float)rand() / RAND_MAX - 0.5f); // Random [-scale/2, scale/2]
    }


    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->update = dense_update;
    layer->downstream_grads = (float*)malloc(batch_size * in_dim * sizeof(float));
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


void dense_forward(Layer* layer, int batch_size) {  
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 
    
    float* weights = layer->weights;
    float* inputs = layer->inputs;
    float* inputs_augmented = denseLayer->inputs_augmented;
    float* outs = layer->outputs;
    int in_dim = denseLayer->in_dim;
    int out_dim = denseLayer->out_dim;

    host_populate_inputs_augmented(inputs, inputs_augmented, batch_size, in_dim);
    host_matrix_multiply(inputs_augmented, weights, outs, batch_size, in_dim + 1, out_dim);
}


void dense_backward(Layer* layer, int batch_size) {
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 
    host_dense_backward(denseLayer, layer, batch_size);
}

void host_dense_backward(DenseLayer* denseLayer, Layer* layer, int batch_size) {

    int in_dim = denseLayer->in_dim;
    int out_dim = denseLayer->out_dim;
    float* weights = layer->weights;
    float* weights_T = denseLayer->weights_T;
    float* weights_grads = layer->weights_grads;
    float* inputs_augmented = denseLayer->inputs_augmented;
    float* inputs_augmented_T = denseLayer->inputs_augmented_T;

    host_transpose(inputs_augmented, inputs_augmented_T, batch_size, in_dim + 1);
    host_matrix_multiply(inputs_augmented_T, layer->upstream_grads, weights_grads, in_dim + 1, batch_size, out_dim);

    host_transpose(weights, weights_T, in_dim + 1, out_dim);
    host_matrix_multiply(layer->upstream_grads, weights_T, inputs_augmented, batch_size, out_dim, in_dim + 1); // reuse of memory allocated for inputs augmented

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_dim; j++) {
            layer->downstream_grads[i * in_dim + j] = inputs_augmented[i * (in_dim + 1) + j];
        }
    }

    if (!USE_MPI) {
        dense_update(layer, batch_size);
    }
}



void dense_update(Layer* layer, int batch_size) {
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 
    float* weights = layer->weights;
    float* weights_grads = layer->weights_grads;
    int in_dim = denseLayer->in_dim;
    int out_dim = denseLayer->out_dim;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < in_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            weights[i * out_dim + j] -= LR * weights_grads[i * out_dim + j] / batch_size;
        }
    }
}