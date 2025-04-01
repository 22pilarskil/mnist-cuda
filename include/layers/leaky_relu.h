#include "../../include/model.h"

typedef struct {
    float coeff;
    int dim;
} LeakyReLU;

Layer* initLeakyReLU(int batch_size, int dim, int coeff, float* inputs);
void leakyReLU_forward(Layer* layer, int batch_size);
void host_leakyReLU_forward(float* inputs, float* outs, int batch_size, int dim, float coeff);
void leakyReLU_backward();
void host_leakyReLU_backward();