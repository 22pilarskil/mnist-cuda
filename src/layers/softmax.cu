#include "../../include/model.h"
#include "../../include/utils.h"
#include "../../include/layers/softmax.h"
#include <stdio.h>


Layer* initSoftmax(int batch_size, int dim, float* inputs) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    Softmax* softmax = (Softmax*)malloc(sizeof(Softmax));
    softmax->dim = dim;

    cudaMallocManaged(&layer->outputs, batch_size * dim * sizeof(float));

    layer->downstream_grads = (float*)malloc(batch_size * dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = softmax;
    layer->type = LAYER_SOFTMAX;
    return layer;
}


void softmax_forward(Layer* layer, int batch_size) {
    Softmax* softmax = (Softmax*)layer->layer_data;  
    host_softmax_forward(layer->inputs, layer->outputs, batch_size, softmax->dim);
}


void host_softmax_forward(float* inputs, float* outs, int batch_size, int dim) {
    float* sums = (float*)calloc(batch_size, sizeof(float));
    #pragma omp parallel 
    for (int i = 0; i < batch_size; i++){
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            sums[i] += inputs[idx];
        }
    }

    #pragma omp parallel
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            outs[idx] += inputs[idx] / (sums[i] + 1e-5);
        }
    }
}

void softmax_backward(Layer* layer) {
    host_softmax_backward(layer->inputs);
}

void host_softmax_backward(float* upstream_grads) {
    // print_matrix(upstream_grads, 64, 10);
}