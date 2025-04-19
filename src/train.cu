#include <stdio.h>
#include "../include/mnist_loader.h"
#include "../include/model.h"
#include "../include/macros.h"
#include "../include/utils.h"
#include <mpi.h>
#include <time.h>
#include <chrono>
#include <iostream>


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
    Model* model = init_model(batch_size);

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

            if (USE_MPI) {

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