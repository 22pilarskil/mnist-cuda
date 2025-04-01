#include "../../include/model.h"

typedef struct {
    float* weights;
    int in_dim;
    int out_dim;
    int dim_with_bias;
} DenseLayer;

void dense_forward(Layer* layer, int batch_size);
void host_dense_forward(float* weights, float* inputs, float* outs, int batch_size, int in_dim, int out_dim);
Layer* initDenseLayer(int batch_size, int in_dim, int out_dim, float* inputs);
void dense_backward();
void host_dense_backward();
