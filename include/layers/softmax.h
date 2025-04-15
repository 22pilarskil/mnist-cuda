#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    int dim;
} Softmax;

void softmax_forward(Layer* layer, int batch_size);
Layer* initSoftmax(int batch_size, int dim, float* inputs);
void softmax_backward(Layer* layer, int batch_size);
void softmax_update(Layer* layer, int batch_size);

void host_softmax_forward(float* inputs, float* outs, int batch_size, int dim);
__global__ void cuda_softmax_forward_kernel(float* inputs, float* outs, int batch_size, int dim);
void cuda_softmax_forward(float* inputs, float* outs, int batch_size, int dim);

void host_softmax_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim);
__global__ void cuda_softmax_backward_kernel(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim);
void cuda_softmax_backward(float* upstream_grads, float* downstream_grads, float* outputs, int batch_size, int dim);


