#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include "../../include/layers/leaky_relu.h"
#include <stdio.h>


Layer* initLeakyReLU(int batch_size, int dim, int coeff, float* inputs) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    LeakyReLU* leakyReLU = (LeakyReLU*)malloc(sizeof(LeakyReLU));
    leakyReLU->coeff = coeff;
    leakyReLU->dim = dim;

    cudaMallocManaged(&layer->outputs, batch_size * dim * sizeof(float));

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
    host_leakyReLU_forward(layer->inputs, layer->outputs, batch_size, leakyReLU->dim, leakyReLU->coeff);

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

void leakyReLU_backward(Layer* layer, int batch_size) {
    LeakyReLU* leakyReLU = (LeakyReLU*)layer->layer_data;
    host_leakyReLU_backward(layer, leakyReLU, batch_size);
}

void host_leakyReLU_backward(Layer* layer, LeakyReLU* leakyReLU, int batch_size) {
    int dim = leakyReLU->dim;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            if (layer->inputs[idx] >= 0) {
                layer->downstream_grads[idx] = layer->upstream_grads[idx];
            } else {
                layer->downstream_grads[idx] = leakyReLU->coeff * layer->upstream_grads[idx];
            }
        }
    }
}

void leakyReLU_update(Layer* layer, int batch_size) {
    
}