#include "../include/model.h"
#include "../include/utils.h"
#include "../include/layers/input_buffer.h"
#include "../include/layers/dense.h"
#include "../include/layers/leaky_relu.h"
#include "../include/layers/softmax.h"
#include "../include/loss/cross_entropy.h"
#include <stdio.h>

void forward(Model* model, float* inputs, uint8_t* targets) {
    for (int i = 0; i < model->n_layers; i++) {
        switch (model->layers[i]->type) {
            case LAYER_INPUT: {
                inputBuffer_forward(model->layers[i], inputs, model->batch_size);
                break;
            }
            case LAYER_DENSE: {
                dense_forward(model->layers[i], model->batch_size);
                break;
            }
            case LAYER_LEAKY_RELU: {
                leakyReLU_forward(model->layers[i], model->batch_size);
                break;
            }
            case LAYER_SOFTMAX: {
                softmax_forward(model->layers[i], model->batch_size);
                break;
            }
            default: {
                fprintf(stderr, "Error: Unexpected layer %d\n", model->layers[i]->type);
                exit(EXIT_FAILURE);
            }
        }
    }
    switch (model->loss->type) {
        case LOSS_CROSS_ENTROPY: {
            cross_entropy_forward(model->loss, model->batch_size, targets);
            break;
        }
        default: {
            fprintf(stderr, "Error: Unexpected loss %d\n", model->loss->type);
            exit(EXIT_FAILURE);
        }
    }
}

void backward(Model* model) {
    for (int i = model->n_layers-1; i > 0; i--) {
        switch (model->layers[i]->type) {
            case LAYER_DENSE: {
                dense_backward();
                break;
            }
            case LAYER_LEAKY_RELU: {
                leakyReLU_backward();
                break;
            }
            case LAYER_SOFTMAX: {
                softmax_backward(model->layers[i]);
                break;
            }
            default: {
                fprintf(stderr, "Error: Unexpected layer %d\n", model->layers[i]->type);
                exit(EXIT_FAILURE);
            }
        }
    }
}


Model* init_model(int batch_size) {
    Model* model = (Model*)malloc(sizeof(Model));
    model->n_layers = 7;
    model->batch_size = batch_size;
    model->layers[0] = initInputBuffer(batch_size, 784);
    model->layers[1] = initDenseLayer(batch_size, 784, 256, model->layers[0]->outputs); // dense 1
    model->layers[2] = initLeakyReLU(batch_size, 256, 0.1, model->layers[1]->outputs); // relu 1
    model->layers[3] = initDenseLayer(batch_size, 256, 64, model->layers[2]->outputs); // dense 2
    model->layers[4] = initLeakyReLU(batch_size, 64, 0.1, model->layers[3]->outputs); // relu 2
    model->layers[5] = initDenseLayer(batch_size, 64, 10, model->layers[4]->outputs); // dense 3
    model->layers[6] = initSoftmax(batch_size, 10, model->layers[5]->outputs); // softmax
    model->loss = initCrossEntropyLoss(batch_size, 10, model->layers[6]->outputs);

    model->layers[6]->upstream_grads = model->loss->downstream_grads;
    model->layers[5]->upstream_grads = model->layers[6]->downstream_grads;
    model->layers[4]->upstream_grads = model->layers[5]->downstream_grads;
    model->layers[3]->upstream_grads = model->layers[4]->downstream_grads;
    model->layers[2]->upstream_grads = model->layers[3]->downstream_grads;
    model->layers[1]->upstream_grads = model->layers[2]->downstream_grads;

    return model;
}



