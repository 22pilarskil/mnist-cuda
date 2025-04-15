#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include "../../include/layers/softmax.h"
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
    host_softmax_forward(layer->inputs, layer->outputs, batch_size, softmax->dim);
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

void softmax_backward(Layer* layer, int batch_size) {
    host_softmax_backward(layer->upstream_grads, layer->downstream_grads, layer->outputs, batch_size, ((Softmax *)layer->layer_data)->dim);
}

void host_softmax_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            downstream_grads[i * dim + j] = upstream_grads[i * dim + j];
        }
    }
}

void softmax_update(Layer* layer, int batch_size) {
    
}