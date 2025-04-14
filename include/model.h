#pragma once
#include <stdint.h>
#include <stdio.h>

typedef void (*ForwardFunc)(struct Layer* layer, int batch_size);
typedef void (*BackwardFunc)(struct Layer* layer, int batch_size);
typedef void (*UpdateFunc)(struct Layer* layer, int batch_size);
typedef void (*LossFunc)(struct Loss* loss, float* targets);

typedef enum {
    LAYER_DENSE,
    LAYER_LEAKY_RELU,
    LAYER_SOFTMAX,
    LAYER_INPUT,
    LAYER_SIGMOID,
    LAYER_TYPE_COUNT
} LayerType;

typedef enum {
    LOSS_CROSS_ENTROPY,
    LOSS_TYPE_COUNT
} LossType;

typedef struct Layer {
    LayerType type;
    ForwardFunc forward;
    BackwardFunc backward;
    UpdateFunc update;
    void* layer_data;
    float* outputs;
    float* inputs;
    float* upstream_grads;
    float* downstream_grads;
    float* weights_grads;
    float* weights;
    int weights_size;
    const char* name;
} Layer;


typedef struct Loss {
    float* downstream_grads;
    LossType type;
    LossFunc lossFunc;
    int dim;
    float loss;
    float accuracy;
    float* inputs;
} Loss;
 
typedef struct {
    int n_layers;
    int batch_size;
    Layer* input_buffer;
    Layer* layers[6];
    Loss* loss;
    int broadcast_weights_size;
    float* broadcast_weights_grads;
} Model;

Model* init_model(int batch_size);

void forward(Model* model, float* inputs, uint8_t* targets);
void backward(Model* model);
void update(Model* model);
