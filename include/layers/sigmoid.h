#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    int dim;
} Sigmoid;

void sigmoid_forward(Layer* layer, int batch_size);
void host_sigmoid_forward(float* inputs, float* outs, int batch_size, int dim);
Layer* initSigmoid(int batch_size, int dim, float* inputs);
void sigmoid_backward(Layer* layer, int batch_size);
void host_sigmoid_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim);
void sigmoid_update(Layer* layer, int batch_size);
