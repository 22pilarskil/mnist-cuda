#include "../../include/model.h"
#include "../../include/utils.h"
#include "../../include/layers/dense.h"
#include <stdio.h>


Layer* initDenseLayer(int batch_size, int in_dim, int out_dim, float* inputs) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    DenseLayer* denseLayer = (DenseLayer*)malloc(sizeof(DenseLayer));
    denseLayer->in_dim = in_dim;
    denseLayer->out_dim = out_dim;

    int dim_with_bias = in_dim + 1;
    denseLayer->dim_with_bias = dim_with_bias;

    cudaMallocManaged(&layer->outputs, batch_size * out_dim * sizeof(float*));
    cudaMallocManaged(&denseLayer->weights, dim_with_bias * out_dim * sizeof(float));
    cudaMemset(denseLayer->weights, 0, dim_with_bias * out_dim * sizeof(float));

    layer->downstream_grads = (float*)malloc(batch_size * in_dim * out_dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = denseLayer;
    layer->type = LAYER_DENSE;
    return layer;
}


void dense_forward(Layer* layer, int batch_size) {  
    DenseLayer* denseLayer = (DenseLayer*)layer->layer_data;
    print_matrix(layer->inputs, batch_size, denseLayer->in_dim);      
    host_matrix_multiply(denseLayer->weights, layer->inputs, layer->outputs, 
        batch_size, denseLayer->in_dim, denseLayer->out_dim);
}


void host_dense_forward(float* weights, float* inputs, float* outs, int batch_size, int in_dim, int out_dim) {
    host_matrix_multiply(weights, inputs, outs, batch_size, in_dim, out_dim);
}

void dense_backward() {

}

void host_dense_backward() {

}