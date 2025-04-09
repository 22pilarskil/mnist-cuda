#include "../../include/model.h"
#include "../../include/utils.h"
#include "../../include/loss/cross_entropy.h"
#include <stdio.h>
#include <math.h>

Loss* initCrossEntropyLoss(int batch_size, int dim, float* inputs) {
    Loss* loss = (Loss*)malloc(sizeof(Loss));
    loss->downstream_grads = (float*)malloc(batch_size * dim * sizeof(float));
    loss->type = LOSS_CROSS_ENTROPY;
    loss->dim = dim;
    loss->inputs = inputs;
    return loss;
}

void cross_entropy_forward(Loss* loss, int batch_size, uint8_t* targets) {
    host_cross_entropy(loss, batch_size, targets);
}

void host_cross_entropy(Loss* loss, int batch_size, uint8_t* targets) {

    float total_loss = 0;
    #pragma omp parallel
    for (int i = 0; i < batch_size; i++) {
        int non_zero_count = 0;
        for (int j = 0; j < loss->dim; j++) {
            int idx = i * loss->dim + j;
            float predicted = loss->inputs[idx];
            if (targets[i] == j) {
                loss->downstream_grads[idx] = predicted - 1.0f;
                total_loss += -log(loss->inputs[idx] + 1e-5);
                non_zero_count++;
            } else {
                loss->downstream_grads[idx] = predicted;
            }
        }
        if (non_zero_count > 1) {
            fprintf(stderr, "Error: non_zero_count was greater than 1\n");
            exit(EXIT_FAILURE);
        }
    }
    printf("TOTAL LOSS: %f\n", total_loss / 64);
}