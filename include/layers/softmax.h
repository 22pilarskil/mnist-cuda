#include "../../include/model.h"

typedef struct {
    int dim;
} Softmax;

void softmax_forward(Layer* layer, int batch_size);
void host_softmax_forward(float* inputs, float* outs, int batch_size, int dim);
Layer* initSoftmax(int batch_size, int dim, float* inputs);
void softmax_backward(Layer* layer, int batch_size);
void host_softmax_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim);
void softmax_update(Layer* layer, int batch_size);
