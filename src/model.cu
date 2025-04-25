#include "../include/model.h"
#include "../include/macros.h"
#include "../include/utils.h"
#include <stdio.h>

float* forward(Model* model, float* inputs, uint8_t* targets) {
    inputBuffer_forward(model->input_buffer, inputs, model->batch_size);
    for (int i = 0; i < model->n_layers; i++) {
        if (model->layers[i]->type >= LAYER_TYPE_COUNT) {
            printf("Invalid layer type: %d\n", model->layers[i]->type);
            exit(EXIT_FAILURE);
        }
        model->layers[i]->forward(model->layers[i], model->batch_size);
    }
    if (model->loss != NULL) {
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
        return model->loss->inputs; 
    }
    return NULL;
}

void backward(Model* model) {
    int offset = model->broadcast_weights_size;
    for (int i = model->n_layers-1; i >= 0; i--) {
        if (model->layers[i]->type >= LAYER_TYPE_COUNT) {
            printf("Invalid layer type: %d\n", model->layers[i]->type);
            exit(EXIT_FAILURE);
        }
        model->layers[i]->backward(model->layers[i], model->batch_size);
        if (USE_MPI_WEIGHT_SHARING) {
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




