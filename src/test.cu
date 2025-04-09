#include "../include/utils.h"
#include "../include/layers/softmax.h"
#include "../include/layers/leaky_relu.h"
#include "../include/layers/sigmoid.h"
#include "../include/layers/dense.h"
#include "../include/loss/cross_entropy.h"
#include <stdio.h>
#include <math.h> // For expf, etc.

void test_matrix_multiply() {
    printf("Testing host_matrix_multiply: ");
    float A[2][3] = {{1, 2, 3}, {4, 5, 6}};
    float B[3][2] = {{7, 8}, {9, 10}, {11, 12}};
    float C[2][2] = {{0}};
    float expected_C[2][2] = {{58, 64}, {139, 154}};
    host_matrix_multiply(&A[0][0], &B[0][0], &C[0][0], 2, 3, 2);
    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(C[i][j] - expected_C[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_transpose() {
    printf("Testing host_transpose: ");
    float A[3][2] = {{7, 8}, {9, 10}, {11, 12}};
    float A_T[2][3] = {{0}};
    float expected_A_T[2][3] = {{7, 9, 11}, {8, 10, 12}};
    host_transpose(&A[0][0], &A_T[0][0], 3, 2);
    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {  // Fixed loop bound to 3
            if (fabs(A_T[i][j] - expected_A_T[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_softmax() {
    printf("Testing host_softmax_forward: ");
    float softmax_in[2][3] = {{1.0, 2.0, 3.0}, {0.0, 2.0, -1.0}};
    float softmax_out[2][3] = {{0}};
    float expected_out[2][3] = {{0.09, 0.24, 0.67}, {0.11, 0.84, 0.04}};
    host_softmax_forward(&softmax_in[0][0], &softmax_out[0][0], 2, 3);
    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(softmax_out[i][j] - expected_out[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_leaky_relu() {
    printf("Testing host_leakyReLU_forward: ");
    float relu_in[2][2] = {{1.0, -2.0}, {0.0, -0.5}};
    float relu_out[2][2] = {{0}};
    float expected_relu_out[2][2] = {{1.0, -0.02}, {0.0, -0.005}};
    float coeff = 0.01;
    host_leakyReLU_forward(&relu_in[0][0], &relu_out[0][0], 2, 2, coeff);
    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(relu_out[i][j] - expected_relu_out[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_sigmoid() {
    printf("Testing host_sigmoid_forward: ");
    float sigmoid_in[2][2] = {{0.0, 1.0}, {-1.0, 2.0}};
    float sigmoid_out[2][2] = {{0}};
    float expected_sigmoid_out[2][2] = {{0.5, 0.73}, {0.27, 0.88}};
    host_sigmoid_forward(&sigmoid_in[0][0], &sigmoid_out[0][0], 2, 2);
    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(sigmoid_out[i][j] - expected_sigmoid_out[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_dense() {
    printf("Testing host_dense_forward: ");
    float dense_weights[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {0.5, 0.5}};
    float dense_in[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    float dense_inputs_augmented[2][3];
    float dense_out[2][2] = {{0}};
    float expected_dense_out[2][2] = {{7.5, 10.5}, {15.5, 22.5}};
    host_dense_forward(&dense_weights[0][0], &dense_in[0][0], &dense_inputs_augmented[0][0], &dense_out[0][0], 2, 2, 2);
    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(dense_out[i][j] - expected_dense_out[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_dense_backward() {
    printf("Testing host_dense_backward: ");
    float dense_weights[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {0.5, 0.5}};
    float dense_in[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    float dense_inputs_augmented[2][3];
    float dense_out[2][2] = {{0}};
    host_dense_forward(&dense_weights[0][0], &dense_in[0][0], &dense_inputs_augmented[0][0], &dense_out[0][0], 2, 2, 2);

    Layer dense_layer;
    DenseLayer dense_data = {0};
    dense_layer.layer_data = &dense_data;
    dense_data.in_dim = 2;
    dense_data.out_dim = 2;
    dense_data.id = 1;
    dense_data.weights = &dense_weights[0][0];
    float weights_T[2][3];
    dense_data.weights_T = &weights_T[0][0];
    float weights_grad[3][2] = {{0}};
    dense_data.weights_grad = &weights_grad[0][0];
    dense_data.inputs_augmented = &dense_inputs_augmented[0][0];
    float inputs_augmented_T[3][2];
    dense_data.inputs_augmented_T = &inputs_augmented_T[0][0];
    float upstream_grads[2][2] = {{0.1, 0.2}, {0.3, 0.4}};
    dense_layer.upstream_grads = &upstream_grads[0][0];
    float downstream_grads[2][2];
    dense_layer.downstream_grads = &downstream_grads[0][0];

    float expected_weights_grad[3][2] = {{1.0, 1.4}, {1.4, 2.0}, {0.4, 0.6}};
    float expected_downstream_grads[2][2] = {{0.5, 1.1}, {1.1, 2.5}};

    host_dense_backward(&dense_data, &dense_layer, 2);

    char* result = "PASS";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(weights_grad[i][j] - expected_weights_grad[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(downstream_grads[i][j] - expected_downstream_grads[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_leaky_relu_backward() {
    printf("Testing host_leakyReLU_backward: ");
    Layer layer;
    LeakyReLU leaky_data = {0};
    layer.layer_data = &leaky_data;
    leaky_data.dim = 2;
    leaky_data.coeff = 0.01;

    float inputs[2][2] = {{1.0, -2.0}, {0.0, -0.5}};
    float outputs[2][2];
    float upstream_grads[2][2] = {{0.1, 0.2}, {0.3, 0.4}};
    float downstream_grads[2][2];
    float expected_downstream_grads[2][2] = {{0.1, 0.002}, {0.3, 0.004}};

    layer.inputs = &inputs[0][0];
    layer.outputs = &outputs[0][0];
    layer.upstream_grads = &upstream_grads[0][0];
    layer.downstream_grads = &downstream_grads[0][0];

    host_leakyReLU_forward(&inputs[0][0], &outputs[0][0], 2, 2, leaky_data.coeff);
    host_leakyReLU_backward(&layer, &leaky_data, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(downstream_grads[i][j] - expected_downstream_grads[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_sigmoid_backward() {
    printf("Testing host_sigmoid_backward: ");
    Layer layer;
    Sigmoid sigmoid_data = {0};
    layer.layer_data = &sigmoid_data;
    sigmoid_data.dim = 2;

    float inputs[2][2] = {{0.0, 1.0}, {-1.0, 2.0}};
    float outputs[2][2];
    float upstream_grads[2][2] = {{0.1, 0.2}, {0.3, 0.4}};
    float downstream_grads[2][2];
    float expected_downstream_grads[2][2] = {{0.025, 0.039}, {0.059, 0.042}};

    layer.inputs = &inputs[0][0];
    layer.outputs = &outputs[0][0];
    layer.upstream_grads = &upstream_grads[0][0];
    layer.downstream_grads = &downstream_grads[0][0];

    host_sigmoid_forward(&inputs[0][0], &outputs[0][0], 2, 2);
    host_sigmoid_backward(&upstream_grads[0][0], &downstream_grads[0][0], &outputs[0][0], 2, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(downstream_grads[i][j] - expected_downstream_grads[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

void test_cross_entropy() {
    printf("Testing host_cross_entropy: ");
    Loss loss = {0};
    loss.dim = 3;
    float inputs[2][3] = {{0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}};
    loss.inputs = &inputs[0][0];
    float downstream_grads[2][3];
    loss.downstream_grads = &downstream_grads[0][0];
    uint8_t targets[2] = {2, 1}; // Target class indices
    float expected_downstream_grads[2][3] = {{0.1, 0.2, -0.3}, {0.3, -0.6, 0.3}};

    host_cross_entropy(&loss, 2, targets);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(downstream_grads[i][j] - expected_downstream_grads[i][j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);
}

int main() {
    test_matrix_multiply();
    test_transpose();
    test_softmax();
    test_leaky_relu();
    test_sigmoid();
    test_dense();
    test_dense_backward();
    test_leaky_relu_backward();
    test_sigmoid_backward();
    test_cross_entropy();

    return 0;
}