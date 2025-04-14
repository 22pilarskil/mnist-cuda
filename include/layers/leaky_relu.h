#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    float coeff;
    int dim;
} LeakyReLU;

Layer* initLeakyReLU(int batch_size, int dim, int coeff, float* inputs);
void leakyReLU_forward(Layer* layer, int batch_size);
void host_leakyReLU_forward(float* inputs, float* outs, int batch_size, int dim, float coeff);
void leakyReLU_backward(Layer* layer, int batch_size);
void host_leakyReLU_backward(Layer* layer, LeakyReLU* leakyReLU, int batch_size);
void leakyReLU_update(Layer* layer, int batch_size);

