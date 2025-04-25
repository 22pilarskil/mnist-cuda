#include <stdio.h>
#include "../include/mnist_loader.h"
#include "../include/model.h"
#include "../include/macros.h"
#include "../include/utils.h"
#include <mpi.h>
#include <time.h>
#include <chrono>
#include <iostream>

static Model* init_model(int batch_size, int rank) {
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

static Model* init_model_parallel(int batch_size, int rank) {

    Model* model = (Model*)malloc(sizeof(Model));
    if (rank % 2 == 0) {

    }
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

#define EPOCHS 100

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    int device_id = rank % nDevices;
    cudaSetDevice(device_id);

    uint32_t batch_size = 64;
    MNISTData data = load_mnist(batch_size, rank);
    Model* model = init_model(batch_size, rank);

    float* batch_images = (float*)malloc(batch_size * IMAGE_SIZE * sizeof(float));
    uint8_t* batch_labels = (uint8_t*)malloc(batch_size * sizeof(float));
    if (!batch_images || !batch_labels) {
        perror("Failed to allocate batch memory");
        free_mnist(&data);
        exit(1);
    }

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    // Iterate over training batches
    if (rank == 0) {
        make_results_dir();
    }
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        auto start = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < data.train.num_batches; i++) {

            load_batch(&data, 1, i, batch_images, batch_labels);

            forward(model, batch_images, batch_labels);
            backward(model);

            MPI_Barrier(MPI_COMM_WORLD);

            if (USE_MPI_WEIGHT_SHARING) {

                MPI_Allreduce(MPI_IN_PLACE, model->broadcast_weights_grads, model->broadcast_weights_size / sizeof(float), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                if (USE_CUDA) {
                    cuda_multiply(model->broadcast_weights_grads, model->broadcast_weights_size / sizeof(float), 1. / size);
                } else {
                    host_multiply(model->broadcast_weights_grads, model->broadcast_weights_size / sizeof(float), 1. / size);
                }
                update(model);

            }
        }

        float total_loss = 0;
        float total_accuracy = 0;
        for (uint32_t i = 0; i < data.test.num_batches; i++) {

            load_batch(&data, 0, i, batch_images, batch_labels);
            forward(model, batch_images, batch_labels);
            total_loss += model->loss->loss;
            total_accuracy += model->loss->accuracy;
            
        }
        MPI_Barrier(MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time_spent = end - start;
        float avg_accuracy = total_accuracy / data.test.num_batches;
        float avg_loss = total_loss / data.test.num_batches;
        printf("EPOCH: %d | Avg Accuracy: %f | Avg Loss: %f | Time elapsed: %f | Rank: %d | GPU ID: %d\n", epoch, avg_accuracy, avg_loss, time_spent.count(), rank, device_id);

        write_results(epoch, avg_accuracy, avg_loss, rank, time_spent.count());
        fflush(stdout);
    }


    FILE* out = fopen("predictions.txt", "w");
    if (out == NULL) {
        perror("Failed to open file");
        return 1;
    }
    load_batch(&data, 0, 0, batch_images, batch_labels);
    float* preds = forward(model, batch_images, batch_labels);
    for (int i = 0; i < batch_size; i++) {
        float max = -INFINITY;
        int label = 0;
        for (int j = 0; j < 10; j++) {
            if (preds[i * 10 + j] > max) {
                max = preds[i * 10 + j];
                label = j;
            }
        }
        print_grayscale_image(&batch_images[i * 28 * 28], 28, 28, out);
        fprintf(out, "Predicted: %d, Actual: %d\n", label, batch_labels[i]);
    }
    print_matrix(preds, 64, 10);
        

    free(batch_images);
    free(batch_labels);
    free_mnist(&data);

    MPI_Finalize();
    return 0;
}
