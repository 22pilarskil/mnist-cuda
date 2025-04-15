#include "../include/utils.h"
#include "../include/layers/softmax.h"
#include "../include/layers/leaky_relu.h"
#include "../include/layers/sigmoid.h"
#include "../include/layers/dense.h"
#include "../include/loss/cross_entropy.h"
#include <stdio.h>
#include <math.h>

void test_host_matrix_multiply() {
    printf("Testing host_matrix_multiply: ");
    float *A, *B, *C;
    MALLOC(&A, 2 * 3 * sizeof(float));
    MALLOC(&B, 3 * 2 * sizeof(float));
    MALLOC(&C, 2 * 2 * sizeof(float));

    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;

    B[0] = 7; B[1] = 8;
    B[2] = 9; B[3] = 10;
    B[4] = 11; B[5] = 12;

    float *expected_C;
    MALLOC(&expected_C, 2 * 2 * sizeof(float));
    expected_C[0] = 58; expected_C[1] = 64;
    expected_C[2] = 139; expected_C[3] = 154;

    host_matrix_multiply(A, B, C, 2, 3, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(C[i * 2 + j] - expected_C[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(A);
    FREE(B);
    FREE(C);
    FREE(expected_C);
}

void test_cuda_matrix_multiply() {
    printf("Testing cuda_matrix_multiply: ");
    float *a, *b, *c;
    MALLOC(&a, 2 * 3 * sizeof(float));
    MALLOC(&b, 3 * 2 * sizeof(float));
    MALLOC(&c, 2 * 2 * sizeof(float));

    a[0] = 1; a[1] = 2; a[2] = 3;
    a[3] = 4; a[4] = 5; a[5] = 6;

    b[0] = 7; b[1] = 8;
    b[2] = 9; b[3] = 10;
    b[4] = 11; b[5] = 12;

    float *expected_C;
    cudaMallocManaged(&expected_C, 2 * 2 * sizeof(float));
    expected_C[0] = 58; expected_C[1] = 64;
    expected_C[2] = 139; expected_C[3] = 154;

    cuda_matrix_multiply(a, b, c, 2, 3, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(c[i * 2 + j] - expected_C[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(a);
    FREE(b);
    FREE(c);
    FREE(expected_C);
}

void test_host_transpose() {
    printf("Testing host_transpose: ");
    float *A, *A_T;
    MALLOC(&A, 3 * 2 * sizeof(float));
    MALLOC(&A_T, 2 * 3 * sizeof(float));

    A[0] = 7; A[1] = 8;
    A[2] = 9; A[3] = 10;
    A[4] = 11; A[5] = 12;

    float *expected_A_T;
    MALLOC(&expected_A_T, 2 * 3 * sizeof(float));
    expected_A_T[0] = 7; expected_A_T[1] = 9; expected_A_T[2] = 11;
    expected_A_T[3] = 8; expected_A_T[4] = 10; expected_A_T[5] = 12;

    host_transpose(A, A_T, 3, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(A_T[i * 3 + j] - expected_A_T[i * 3 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(A);
    FREE(A_T);
    FREE(expected_A_T);
}

void test_cuda_transpose() {
    printf("Testing cuda_transpose: ");
    float *A, *A_T;
    MALLOC(&A, 3 * 2 * sizeof(float));
    MALLOC(&A_T, 2 * 3 * sizeof(float));

    A[0] = 7; A[1] = 8;
    A[2] = 9; A[3] = 10;
    A[4] = 11; A[5] = 12;

    float *expected_A_T;
    MALLOC(&expected_A_T, 2 * 3 * sizeof(float));
    expected_A_T[0] = 7; expected_A_T[1] = 9; expected_A_T[2] = 11;
    expected_A_T[3] = 8; expected_A_T[4] = 10; expected_A_T[5] = 12;

    cuda_transpose(A, A_T, 3, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(A_T[i * 3 + j] - expected_A_T[i * 3 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(A);
    FREE(A_T);
    FREE(expected_A_T);
}

void test_softmax() {
    printf("Testing softmax_forward: ");
    float *softmax_in, *softmax_out;
    MALLOC(&softmax_in, 2 * 3 * sizeof(float));
    MALLOC(&softmax_out, 2 * 3 * sizeof(float));

    softmax_in[0] = 1.0; softmax_in[1] = 2.0; softmax_in[2] = 3.0;
    softmax_in[3] = 0.0; softmax_in[4] = 2.0; softmax_in[5] = -1.0;

    float *expected_out;
    MALLOC(&expected_out, 2 * 3 * sizeof(float));
    expected_out[0] = 0.09; expected_out[1] = 0.24; expected_out[2] = 0.67;
    expected_out[3] = 0.11; expected_out[4] = 0.84; expected_out[5] = 0.04;

    Layer layer;
    Softmax softmax_data = {0};
    softmax_data.dim = 3;
    layer.layer_data = &softmax_data;
    layer.inputs = softmax_in;
    layer.outputs = softmax_out;
    layer.forward = softmax_forward;

    layer.forward(&layer, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(softmax_out[i * 3 + j] - expected_out[i * 3 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(softmax_in);
    FREE(softmax_out);
    FREE(expected_out);
}

void test_leaky_relu() {
    printf("Testing leakyReLU_forward: ");
    float *relu_in, *relu_out;
    MALLOC(&relu_in, 2 * 2 * sizeof(float));
    MALLOC(&relu_out, 2 * 2 * sizeof(float));

    relu_in[0] = 1.0; relu_in[1] = -2.0;
    relu_in[2] = 0.0; relu_in[3] = -0.5;

    float *expected_relu_out;
    MALLOC(&expected_relu_out, 2 * 2 * sizeof(float));
    expected_relu_out[0] = 1.0; expected_relu_out[1] = -0.02;
    expected_relu_out[2] = 0.0; expected_relu_out[3] = -0.005;

    float coeff = 0.01;

    Layer layer;
    LeakyReLU leaky_data = {0};
    leaky_data.dim = 2;
    leaky_data.coeff = coeff;
    layer.layer_data = &leaky_data;
    layer.inputs = relu_in;
    layer.outputs = relu_out;
    layer.forward = leakyReLU_forward;

    layer.forward(&layer, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(relu_out[i * 2 + j] - expected_relu_out[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(relu_in);
    FREE(relu_out);
    FREE(expected_relu_out);
}

void test_sigmoid() {
    printf("Testing sigmoid_forward: ");
    float *sigmoid_in, *sigmoid_out;
    MALLOC(&sigmoid_in, 2 * 2 * sizeof(float));
    MALLOC(&sigmoid_out, 2 * 2 * sizeof(float));

    sigmoid_in[0] = 0.0; sigmoid_in[1] = 1.0;
    sigmoid_in[2] = -1.0; sigmoid_in[3] = 2.0;

    float *expected_sigmoid_out;
    MALLOC(&expected_sigmoid_out, 2 * 2 * sizeof(float));
    expected_sigmoid_out[0] = 0.5; expected_sigmoid_out[1] = 0.73;
    expected_sigmoid_out[2] = 0.27; expected_sigmoid_out[3] = 0.88;

    Layer layer;
    Sigmoid sigmoid_data = {0};
    sigmoid_data.dim = 2;
    layer.layer_data = &sigmoid_data;
    layer.inputs = sigmoid_in;
    layer.outputs = sigmoid_out;
    layer.forward = sigmoid_forward;

    layer.forward(&layer, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(sigmoid_out[i * 2 + j] - expected_sigmoid_out[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(sigmoid_in);
    FREE(sigmoid_out);
    FREE(expected_sigmoid_out);
}

void test_dense() {
    printf("Testing dense_forward: ");
    float *dense_weights, *dense_in, *dense_out, *inputs_augmented;
    MALLOC(&dense_weights, 3 * 2 * sizeof(float));
    MALLOC(&dense_in, 2 * 2 * sizeof(float));
    MALLOC(&dense_out, 2 * 2 * sizeof(float));
    MALLOC(&inputs_augmented, 2 * 3 * sizeof(float));

    dense_weights[0] = 1.0; dense_weights[1] = 2.0;
    dense_weights[2] = 3.0; dense_weights[3] = 4.0;
    dense_weights[4] = 0.5; dense_weights[5] = 0.5;

    dense_in[0] = 1.0; dense_in[1] = 2.0;
    dense_in[2] = 3.0; dense_in[3] = 4.0;

    float *expected_out;
    MALLOC(&expected_out, 2 * 2 * sizeof(float));
    expected_out[0] = 7.5; expected_out[1] = 10.5;
    expected_out[2] = 15.5; expected_out[3] = 22.5;

    Layer layer;
    DenseLayer dense_data = {0};
    dense_data.in_dim = 2;
    dense_data.out_dim = 2;
    dense_data.inputs_augmented = inputs_augmented;
    layer.layer_data = &dense_data;
    layer.weights = dense_weights;
    layer.inputs = dense_in;
    layer.outputs = dense_out;
    layer.forward = dense_forward;

    layer.forward(&layer, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(dense_out[i * 2 + j] - expected_out[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(dense_weights);
    FREE(dense_in);
    FREE(dense_out);
    FREE(inputs_augmented);
    FREE(expected_out);
}

void test_dense_backward() {
    printf("Testing dense_backward: ");
    float *dense_weights, *dense_in, *dense_out, *inputs_augmented;
    float *weights_T, *weights_grad, *inputs_augmented_T, *upstream_grads, *downstream_grads;
    MALLOC(&dense_weights, 3 * 2 * sizeof(float));
    MALLOC(&dense_in, 2 * 2 * sizeof(float));
    MALLOC(&dense_out, 2 * 2 * sizeof(float));
    MALLOC(&inputs_augmented, 2 * 3 * sizeof(float));
    MALLOC(&weights_T, 2 * 3 * sizeof(float));
    MALLOC(&weights_grad, 3 * 2 * sizeof(float));
    MALLOC(&inputs_augmented_T, 3 * 2 * sizeof(float));
    MALLOC(&upstream_grads, 2 * 2 * sizeof(float));
    MALLOC(&downstream_grads, 2 * 2 * sizeof(float));

    dense_weights[0] = 1.0; dense_weights[1] = 2.0;
    dense_weights[2] = 3.0; dense_weights[3] = 4.0;
    dense_weights[4] = 0.5; dense_weights[5] = 0.5;

    dense_in[0] = 1.0; dense_in[1] = 2.0;
    dense_in[2] = 3.0; dense_in[3] = 4.0;

    upstream_grads[0] = 0.1; upstream_grads[1] = 0.2;
    upstream_grads[2] = 0.3; upstream_grads[3] = 0.4;

    float *expected_weights_grad, *expected_downstream_grads;
    MALLOC(&expected_weights_grad, 3 * 2 * sizeof(float));
    MALLOC(&expected_downstream_grads, 2 * 2 * sizeof(float));
    expected_weights_grad[0] = 1.0; expected_weights_grad[1] = 1.4;
    expected_weights_grad[2] = 1.4; expected_weights_grad[3] = 2.0;
    expected_weights_grad[4] = 0.4; expected_weights_grad[5] = 0.6;
    expected_downstream_grads[0] = 0.5; expected_downstream_grads[1] = 1.1;
    expected_downstream_grads[2] = 1.1; expected_downstream_grads[3] = 2.5;

    Layer layer;
    DenseLayer dense_data = {0};
    dense_data.in_dim = 2;
    dense_data.out_dim = 2;
    dense_data.id = 1;
    dense_data.inputs_augmented = inputs_augmented;
    dense_data.weights_T = weights_T;
    dense_data.inputs_augmented_T = inputs_augmented_T;
    layer.layer_data = &dense_data;
    layer.weights = dense_weights;
    layer.weights_grads = weights_grad;
    layer.inputs = dense_in;
    layer.outputs = dense_out;
    layer.upstream_grads = upstream_grads;
    layer.downstream_grads = downstream_grads;
    layer.forward = dense_forward;
    layer.backward = dense_backward;

    layer.forward(&layer, 2);
    layer.backward(&layer, 2);

    char* result = "PASS";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(weights_grad[i * 2 + j] - expected_weights_grad[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(downstream_grads[i * 2 + j] - expected_downstream_grads[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(dense_weights);
    FREE(dense_in);
    FREE(dense_out);
    FREE(inputs_augmented);
    FREE(weights_T);
    FREE(weights_grad);
    FREE(inputs_augmented_T);
    FREE(upstream_grads);
    FREE(downstream_grads);
    FREE(expected_weights_grad);
    FREE(expected_downstream_grads);
}

void test_leaky_relu_backward() {
    printf("Testing leakyReLU_backward: ");
    float *inputs, *outputs, *upstream_grads, *downstream_grads;
    MALLOC(&inputs, 2 * 2 * sizeof(float));
    MALLOC(&outputs, 2 * 2 * sizeof(float));
    MALLOC(&upstream_grads, 2 * 2 * sizeof(float));
    MALLOC(&downstream_grads, 2 * 2 * sizeof(float));

    inputs[0] = 1.0; inputs[1] = -2.0;
    inputs[2] = 0.0; inputs[3] = -0.5;

    upstream_grads[0] = 0.1; upstream_grads[1] = 0.2;
    upstream_grads[2] = 0.3; upstream_grads[3] = 0.4;

    float *expected_downstream_grads;
    MALLOC(&expected_downstream_grads, 2 * 2 * sizeof(float));
    expected_downstream_grads[0] = 0.1; expected_downstream_grads[1] = 0.002;
    expected_downstream_grads[2] = 0.3; expected_downstream_grads[3] = 0.004;

    Layer layer;
    LeakyReLU leaky_data = {0};
    leaky_data.dim = 2;
    leaky_data.coeff = 0.01;
    layer.layer_data = &leaky_data;
    layer.inputs = inputs;
    layer.outputs = outputs;
    layer.upstream_grads = upstream_grads;
    layer.downstream_grads = downstream_grads;
    layer.forward = leakyReLU_forward;
    layer.backward = leakyReLU_backward;

    layer.forward(&layer, 2);
    host_leakyReLU_backward(&layer, &leaky_data, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(downstream_grads[i * 2 + j] - expected_downstream_grads[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(inputs);
    FREE(outputs);
    FREE(upstream_grads);
    FREE(downstream_grads);
    FREE(expected_downstream_grads);
}

void test_sigmoid_backward() {
    printf("Testing sigmoid_backward: ");
    float *inputs, *outputs, *upstream_grads, *downstream_grads;
    MALLOC(&inputs, 2 * 2 * sizeof(float));
    MALLOC(&outputs, 2 * 2 * sizeof(float));
    MALLOC(&upstream_grads, 2 * 2 * sizeof(float));
    MALLOC(&downstream_grads, 2 * 2 * sizeof(float));

    inputs[0] = 0.0; inputs[1] = 1.0;
    inputs[2] = -1.0; inputs[3] = 2.0;

    upstream_grads[0] = 0.1; upstream_grads[1] = 0.2;
    upstream_grads[2] = 0.3; upstream_grads[3] = 0.4;

    float *expected_downstream_grads;
    MALLOC(&expected_downstream_grads, 2 * 2 * sizeof(float));
    expected_downstream_grads[0] = 0.025; expected_downstream_grads[1] = 0.039;
    expected_downstream_grads[2] = 0.059; expected_downstream_grads[3] = 0.042;

    Layer layer;
    Sigmoid sigmoid_data = {0};
    sigmoid_data.dim = 2;
    layer.layer_data = &sigmoid_data;
    layer.inputs = inputs;
    layer.outputs = outputs;
    layer.upstream_grads = upstream_grads;
    layer.downstream_grads = downstream_grads;
    layer.forward = sigmoid_forward;
    layer.backward = sigmoid_backward;

    layer.forward(&layer, 2);
    host_sigmoid_backward(upstream_grads, downstream_grads, outputs, 2, 2);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (fabs(downstream_grads[i * 2 + j] - expected_downstream_grads[i * 2 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(inputs);
    FREE(outputs);
    FREE(upstream_grads);
    FREE(downstream_grads);
    FREE(expected_downstream_grads);
}

void test_cross_entropy() {
    printf("Testing host_cross_entropy: ");
    Loss loss = {0};
    loss.dim = 3;
    float *inputs, *downstream_grads;
    MALLOC(&inputs, 2 * 3 * sizeof(float));
    MALLOC(&downstream_grads, 2 * 3 * sizeof(float));

    inputs[0] = 0.1; inputs[1] = 0.2; inputs[2] = 0.7;
    inputs[3] = 0.3; inputs[4] = 0.4; inputs[5] = 0.3;

    float *expected_downstream_grads;
    MALLOC(&expected_downstream_grads, 2 * 3 * sizeof(float));
    expected_downstream_grads[0] = 0.1; expected_downstream_grads[1] = 0.2; expected_downstream_grads[2] = -0.3;
    expected_downstream_grads[3] = 0.3; expected_downstream_grads[4] = -0.6; expected_downstream_grads[5] = 0.3;

    uint8_t targets[2] = {2, 1};
    loss.inputs = inputs;
    loss.downstream_grads = downstream_grads;

    host_cross_entropy(&loss, 2, targets);

    char* result = "PASS";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(downstream_grads[i * 3 + j] - expected_downstream_grads[i * 3 + j]) > 0.01) {
                result = "FAIL";
            }
        }
    }
    printf("%s\n", result);

    FREE(inputs);
    FREE(downstream_grads);
    FREE(expected_downstream_grads);
}

int main() {
    test_host_matrix_multiply();
    test_cuda_matrix_multiply();
    test_host_transpose();
    test_cuda_transpose();
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