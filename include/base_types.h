#pragma once

#include <stdint.h>

typedef struct Layer Layer;
typedef struct Loss Loss;

typedef void (*ForwardFunc)(Layer* layer, int batch_size);
typedef void (*BackwardFunc)(Layer* layer, int batch_size);
typedef void (*UpdateFunc)(Layer* layer, int batch_size);
typedef void (*LossFunc)(Loss* loss, float* targets);

typedef enum {
    LAYER_DENSE,
    LAYER_LEAKY_RELU,
    LAYER_SOFTMAX,
    LAYER_INPUT,
    LAYER_SIGMOID,
    LAYER_MPI_SEND_BUFFER,
    LAYER_MPI_RECV_BUFFER,
    LAYER_TYPE_COUNT
} LayerType;

typedef enum {
    LOSS_CROSS_ENTROPY,
    LOSS_TYPE_COUNT
} LossType;

struct Layer {
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
    int id;
    const char* name;
};

struct Loss {
    float* downstream_grads;
    LossType type;
    LossFunc lossFunc;
    int dim;
    float loss;
    float accuracy;
    float* inputs;
};

typedef struct {
    int n_layers;
    int batch_size;
    Layer* input_buffer;
    Layer** layers;
    Loss* loss;
    int broadcast_weights_size;
    float* broadcast_weights_grads;
    int num_machines;
} Model;
