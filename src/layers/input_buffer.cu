#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include "../../include/layers/input_buffer.h"
#include <stdio.h>


Layer* initInputBuffer(int batch_size, int dim) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    InputBuffer* inputBuffer = (InputBuffer*)malloc(sizeof(InputBuffer));

    inputBuffer->dim = dim;
    cudaMallocManaged(&layer->outputs, batch_size * dim * sizeof(float));

    layer->weights_size = 0;
    layer->layer_data = inputBuffer;
    layer->type = LAYER_INPUT;
    return layer;
}


void inputBuffer_forward(Layer* layer, float* inputs, int batch_size) {
    InputBuffer* inputBuffer = (InputBuffer*)layer->layer_data;      
    cudaMemcpy(layer->outputs, inputs, inputBuffer->dim * batch_size * sizeof(float), cudaMemcpyHostToDevice);
}

void inputBuffer_update(Layer* layer, int batch_size) {
    
}