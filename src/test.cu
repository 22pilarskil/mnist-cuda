#include <stdio.h>

typedef struct {
    float** weights;
    float** grads;
    int in_dim;
    int out_dim;
    int dim_with_bias;
    float** inputs;
    float** outputs;
} DenseLayer;

static DenseLayer* initDenseLayer(int batch_size, int in_dim, int out_dim, float** inputs) {
    DenseLayer* denseLayer = (DenseLayer*)malloc(sizeof(DenseLayer));
    denseLayer->in_dim = in_dim;
    denseLayer->out_dim = out_dim;

    int dim_with_bias = in_dim + 1;
    denseLayer->dim_with_bias = dim_with_bias;

    cudaMallocManaged(&denseLayer->grads, (dim_with_bias) * sizeof(float*));
    cudaMallocManaged(&denseLayer->weights, (dim_with_bias) * sizeof(float*));
    for (int i = 0; i < dim_with_bias; i++) {
        cudaMallocManaged(&denseLayer->grads[i], out_dim * sizeof(float));
        cudaMallocManaged(&denseLayer->weights[i], out_dim * sizeof(float));
        cudaMemset(denseLayer->weights[i], 0, out_dim * sizeof(float));
    }

    if (inputs == NULL) {
        cudaMallocManaged(&denseLayer->inputs, batch_size * sizeof(float*));
        for (int i = 0; i < batch_size; i++) {
            cudaMallocManaged(&denseLayer->inputs[i], dim_with_bias * sizeof(float));
        }
    } else {
        denseLayer->inputs = inputs;
    }

    cudaMallocManaged(&denseLayer->outputs, batch_size * sizeof(float*));
    for (int i = 0; i < batch_size; i++) {
        cudaMallocManaged(&denseLayer->outputs[i], out_dim * sizeof(float));
    }

    return denseLayer;
}

int main() {
    DenseLayer* layer1 = initDenseLayer(64, 784, 256, NULL);
    DenseLayer* layer2 = initDenseLayer(64, 256, 10, layer1->outputs);
    printf("%f | %f\n", layer1->outputs[0][0], layer2->inputs[0][0]);
    layer1->outputs[0][0] = 255.;
    printf("%f | %f\n", layer1->outputs[0][0], layer2->inputs[0][0]);

}