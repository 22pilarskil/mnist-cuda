#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include "../../include/loss/cross_entropy.h"
#include <stdio.h>
#include <math.h>

Loss* initCrossEntropyLoss(int batch_size, int dim, float* inputs) {
    Loss* loss = (Loss*)malloc(sizeof(Loss));
    MALLOC(&loss->downstream_grads, batch_size * dim * sizeof(float));
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
    float total_accuracy = 0;
    #pragma omp parallel
    for (int i = 0; i < batch_size; i++) {
        int max_id = 0;
        float max_score = 0;
        for (int j = 0; j < loss->dim; j++) {
            int idx = i * loss->dim + j;
            float predicted = loss->inputs[idx];
            if (predicted > max_score) {
                max_score = predicted;
                max_id = j;
            }
            if (targets[i] == j) {
                loss->downstream_grads[idx] = predicted - 1.0f;
                total_loss += -log(loss->inputs[idx] + 1e-5);
            } else {
                loss->downstream_grads[idx] = predicted;
            }
        }
        total_accuracy += (max_id == targets[i]);
    }
    loss->accuracy = total_accuracy / batch_size;
    loss->loss = total_loss / batch_size;
}