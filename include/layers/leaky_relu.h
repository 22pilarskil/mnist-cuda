#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    float coeff;
    int dim;
} LeakyReLU;

Layer* initLeakyReLU(int batch_size, int dim, int coeff, float* inputs);
void leakyReLU_forward(Layer* layer, int batch_size);
void leakyReLU_backward(Layer* layer, int batch_size);
void leakyReLU_update(Layer* layer, int batch_size);

void host_leakyReLU_forward(float* inputs, float* outs, int batch_size, int dim, float coeff);
__global__ void cuda_leakyReLU_forward_kernel(float* inputs, float* outs, int batch_size, int dim, float coeff);
void cuda_leakyReLU_forward(float* inputs, float* outs, int batch_size, int dim, float coeff);

void host_leakyReLU_backward(float* inputs, float* upstream_grads, float* downstream_grads, float coeff, int batch_size, int dim);
__global__ void cuda_leakyReLU_backward_kernel(float* inputs, float* upstream_grads, float* downstream_grads, float coeff, int batch_size, int dim);
void cuda_leakyReLU_backward(float* inputs, float* upstream_grads, float* downstream_grads, float coeff, int batch_size, int dim);
