#include "../include/model.h"
#include "../include/macros.h"
#include "../include/utils.h"
#include "../include/layers/input_buffer.h"
#include "../include/layers/dense.h"
#include "../include/layers/leaky_relu.h"
#include "../include/layers/softmax.h"
#include "../include/loss/cross_entropy.h"
#include "../include/layers/sigmoid.h"
#include <stdio.h>

void forward(Model* model, float* inputs, uint8_t* targets) {
    inputBuffer_forward(model->input_buffer, inputs, model->batch_size);
    for (int i = 0; i < model->n_layers; i++) {
        if (model->layers[i]->type >= LAYER_TYPE_COUNT) {
            printf("Invalid layer type: %d\n", model->layers[i]->type);
            exit(EXIT_FAILURE);
        }
        model->layers[i]->forward(model->layers[i], model->batch_size);
    }
    switch (model->loss->type) {
        case LOSS_CROSS_ENTROPY: {
            cross_entropy_forward(model->loss, model->batch_size, targets);
            break;
        }
        default: {
            printf("Error: Unexpected loss %d\n", model->loss->type);
            exit(EXIT_FAILURE);
        }
    }
}

void backward(Model* model) {
    int offset = model->broadcast_weights_size;
    for (int i = model->n_layers-1; i >= 0; i--) {
        if (model->layers[i]->type >= LAYER_TYPE_COUNT) {
            printf("Invalid layer type: %d\n", model->layers[i]->type);
            exit(EXIT_FAILURE);
        }
        model->layers[i]->backward(model->layers[i], model->batch_size);
        if (USE_MPI) {
            int weights_size = model->layers[i]->weights_size;
            if (weights_size > 0) {
                offset -= weights_size;
                COPY((void*)model->broadcast_weights_grads + offset, (void*)model->layers[i]->weights_grads, weights_size);
            }
        }
    }
}

void update(Model* model) {
    int offset = model->broadcast_weights_size;
    for (int i = model->n_layers-1; i >= 0; i--) {
        if (model->layers[i]->type >= LAYER_TYPE_COUNT) {
            printf("Invalid layer type: %d\n", model->layers[i]->type);
            exit(EXIT_FAILURE);
        }

        int weights_size = model->layers[i]->weights_size;
        if (weights_size > 0) {
            offset -= weights_size;
            COPY((void*)model->layers[i]->weights_grads, (void*)model->broadcast_weights_grads + offset, weights_size);
        }
        model->layers[i]->update(model->layers[i], model->batch_size);
    }
}


Model* init_model(int batch_size) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->n_layers = 6;
    model->layers = (Layer**)malloc(sizeof(Layer*) * model->n_layers);
    model->batch_size = batch_size;
    model->input_buffer = initInputBuffer(batch_size, 784);
    model->layers[0] = initDenseLayer(batch_size, 784, 256, model->input_buffer->outputs, 1); // dense 1
    model->layers[1] = initLeakyReLU(batch_size, 256, 0.1, model->layers[0]->outputs); // reul 1
    model->layers[2] = initDenseLayer(batch_size, 256, 64, model->layers[1]->outputs, 2); // dense 2
    model->layers[3] = initLeakyReLU(batch_size, 64, 0.1, model->layers[2]->outputs); // relu 2
    model->layers[4] = initDenseLayer(batch_size, 64, 10, model->layers[3]->outputs, 3); // dense 3
    model->layers[5] = initSoftmax(batch_size, 10, model->layers[4]->outputs); // softmax
    model->loss = initCrossEntropyLoss(batch_size, 10, model->layers[5]->outputs);

    model->layers[5]->upstream_grads = model->loss->downstream_grads;
    model->layers[4]->upstream_grads = model->layers[5]->downstream_grads;
    model->layers[3]->upstream_grads = model->layers[4]->downstream_grads;
    model->layers[2]->upstream_grads = model->layers[3]->downstream_grads;
    model->layers[1]->upstream_grads = model->layers[2]->downstream_grads;
    model->layers[0]->upstream_grads = model->layers[1]->downstream_grads;

    int total_size = 0;
    for (int i = 0; i < model->n_layers; i++) {
        total_size += model->layers[i]->weights_size;
    }
    model->broadcast_weights_size = total_size;
    
    MALLOC((void**)&model->broadcast_weights_grads, total_size);

    return model;
}



