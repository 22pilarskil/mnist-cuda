#include "../../include/model.h"
#include "../../include/utils.h"
#include "../../include/layers/dense.h"
#include <stdio.h>
#include <math.h>


Layer* initDenseLayer(int batch_size, int in_dim, int out_dim, float* inputs, int id) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    DenseLayer* denseLayer = (DenseLayer*)malloc(sizeof(DenseLayer));
    denseLayer->in_dim = in_dim;
    denseLayer->out_dim = out_dim;
    denseLayer->id = id;

    cudaMallocManaged(&layer->outputs, batch_size * out_dim * sizeof(float));
    cudaMallocManaged(&denseLayer->inputs_augmented, batch_size * (in_dim + 1) * sizeof(float));
    cudaMallocManaged(&denseLayer->inputs_augmented_T, batch_size * (in_dim + 1) * sizeof(float));
    cudaMallocManaged(&denseLayer->weights, (in_dim + 1) * out_dim * sizeof(float));
    cudaMallocManaged(&denseLayer->weights_grad, (in_dim + 1) * out_dim * sizeof(float));
    cudaMallocManaged(&denseLayer->weights_T, (in_dim + 1) * out_dim * sizeof(float));

    // srand(42);
    // for (int i = 0; i < (in_dim + 1) * out_dim; i++) {
    //     denseLayer->weights[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    // }


    layer->downstream_grads = (float*)malloc(batch_size * in_dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = denseLayer;
    layer->type = LAYER_DENSE;
    return layer;
}

void host_populate_inputs_augmented(float* inputs, float* inputs_augmented, int batch_size, int in_dim) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_dim; j++) {
            inputs_augmented[i * (in_dim + 1) + j] = inputs[i * in_dim + j];
        }
        inputs_augmented[i * (in_dim + 1) + in_dim] = 1.;
    }
}


void dense_forward(Layer* layer, int batch_size) {  
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 
    host_dense_forward(denseLayer->weights, layer->inputs, denseLayer->inputs_augmented, layer->outputs, 
        batch_size, denseLayer->in_dim, denseLayer->out_dim);
}


void host_dense_forward(float* weights, float* inputs, float* inputs_augmented, float* outs, int batch_size, int in_dim, int out_dim) {
    host_populate_inputs_augmented(inputs, inputs_augmented, batch_size, in_dim);
    host_matrix_multiply(inputs_augmented, weights, outs, batch_size, in_dim + 1, out_dim);

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_dim; j++) {
            if (isnan(inputs[i * out_dim + j])) {
                printf("NAN input value at dense layer on forward");
                exit(1);
            }
        }
    }
}

void dense_backward(Layer* layer, int batch_size) {
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data; 
    host_dense_backward(denseLayer, layer, batch_size);
}

void host_dense_backward(DenseLayer* denseLayer, Layer* layer, int batch_size) {

    int in_dim = denseLayer->in_dim;
    int out_dim = denseLayer->out_dim;
    float* weights = denseLayer->weights;
    float* weights_T = denseLayer->weights_T;
    float* weights_grad = denseLayer->weights_grad;
    float* inputs_augmented = denseLayer->inputs_augmented;
    float* inputs_augmented_T = denseLayer->inputs_augmented_T;

    host_transpose(inputs_augmented, inputs_augmented_T, batch_size, in_dim + 1);
    host_matrix_multiply(inputs_augmented_T, layer->upstream_grads, weights_grad, in_dim + 1, batch_size, out_dim);
    apply_grads(weights, weights_grad, 0.01, in_dim + 1, out_dim, denseLayer->id, batch_size);

    host_transpose(weights, weights_T, in_dim + 1, out_dim);
    host_matrix_multiply(layer->upstream_grads, weights_T, inputs_augmented, batch_size, out_dim, in_dim + 1); // reuse of memory allocated for inputs augmented

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_dim; j++) {
            layer->downstream_grads[i * in_dim + j] = inputs_augmented[i * in_dim + j];
        }
    }
}

void apply_grads(float* weights, float* weights_grad, float lr, int in_dim, int out_dim, int id, int batch_size) {
    for (int i = 0; i < in_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            weights[i * out_dim + j] -= lr * weights_grad[i * out_dim + j] / batch_size;
            if (isnan(weights[i * out_dim + j])) {
                printf("NAN weight value at dense layer %d\n", id);
                exit(1);
            }
        }
    }
}
