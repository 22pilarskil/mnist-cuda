#pragma once
#include <stdint.h>

typedef void (*ForwardFunc)(struct Layer* layer, int batch_size);
typedef void (*LossFunc)(struct Loss* loss, float* targets);

typedef enum {
    LAYER_DENSE,
    LAYER_LEAKY_RELU,
    LAYER_SOFTMAX,
    LAYER_INPUT,
} LayerType;

typedef enum {
    LOSS_CROSS_ENTROPY
} LossType;

typedef struct Layer {
    LayerType type;
    ForwardFunc forward;
    void* layer_data;
    float* outputs;
    float* inputs;
    float* upstream_grads;
    float* downstream_grads;
} Layer;


typedef struct Loss {
    float* downstream_grads;
    LossType type;
    LossFunc lossFunc;
    int dim;
    float* inputs;
} Loss;


typedef struct {
    int n_layers;
    int batch_size;
    Layer* layers[7];
    Loss* loss;
} Model;

Model* init_model(int batch_size);

void forward(Model* model, float* inputs, uint8_t* targets);
void backward(Model* model);
