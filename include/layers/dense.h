#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    float* inputs_augmented;
    float* inputs_augmented_T;
    float* weights_T;
    float* weights_grads;
    float* bias;
    int in_dim;
    int out_dim;
    int dim_with_bias;
    int id;
} DenseLayer;

void dense_forward(Layer* layer, int batch_size);
void host_dense_forward(float* weights, float* inputs, float* inputs_augmented, float* outs, int batch_size, int in_dim, int out_dim);
Layer* initDenseLayer(int batch_size, int in_dim, int out_dim, float* inputs, int id);
void dense_backward(Layer* layer, int batch_size);
void host_dense_backward(DenseLayer* denseLayer, Layer* layer, int batch_size);
void apply_grads(float* weights, float* weights_grads, float lr, int in_dim, int out_dim, int id, int batch_size);
void dense_update(Layer* layer, int batch_size);


void host_extract_downstream_grads(float* downstream_grads, float* inputs_augmented, int batch_size, int in_dim);
__global__ void cuda_extract_downstream_grads_kernel(float* downstream_grads, float* inputs_augmented, int batch_size, int in_dim);
void cuda_extract_downstream_grads(float* downstream_grads, float* inputs_augmented, int batch_size, int in_dim);

void host_populate_inputs_augmented(float* inputs, float* inputs_augmented, int batch_size, int in_dim);
__global__ void cuda_populate_inputs_augmented_kernel(float* inputs, float* inputs_augmented, int batch_size, int in_dim);
void cuda_populate_inputs_augmented(float* inputs, float* inputs_augmented, int batch_size, int in_dim);

void host_weight_update(float* weights, float* weights_grads, int batch_size, int in_dim, int out_dim);
__global__ void cuda_weight_update_kernel(float* weights, float* weights_grads, int batch_size, int in_dim, int out_dim);
void cuda_weight_update(float* weights, float* weights_grads, int batch_size, int in_dim, int out_dim);