#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include <stdio.h>
#include <math.h>


Layer* initSigmoid(int batch_size, int dim, float* inputs) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    Sigmoid* sigmoid = (Sigmoid*)malloc(sizeof(Sigmoid));
    sigmoid->dim = dim;

    MALLOC(&layer->outputs, batch_size * dim * sizeof(float));

    layer->forward = sigmoid_forward;
    layer->backward = sigmoid_backward;
    layer->update = sigmoid_update;
    layer->weights_size = 0;
    MALLOC(&layer->downstream_grads, batch_size * dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = sigmoid;
    layer->type = LAYER_SIGMOID;
    return layer;
}


void sigmoid_forward(Layer* layer, int batch_size) {
    Sigmoid* sigmoid = (Sigmoid*)layer->layer_data;  
    host_sigmoid_forward(layer->inputs, layer->outputs, batch_size, sigmoid->dim);
}


void host_sigmoid_forward(float* inputs, float* outs, int batch_size, int dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            outs[idx] = 1.0f / (1.0f + expf(-inputs[idx]));
        }
    }
}

void sigmoid_backward(Layer* layer, int batch_size) {
    Sigmoid* sigmoid = (Sigmoid *)layer->layer_data;
    host_sigmoid_backward(layer->upstream_grads, layer->downstream_grads, layer->outputs, batch_size, ((Sigmoid *)layer->layer_data)->dim);
}

void host_sigmoid_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            downstream_grads[idx] = outputs[idx] * (1 - outputs[idx]) * upstream_grads[idx];
        }
    }
}

void sigmoid_update(Layer* layer, int batch_size) {
    
}