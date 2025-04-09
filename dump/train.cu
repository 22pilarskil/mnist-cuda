#include <stdio.h>
#include "../include/mnist_loader.h"
#include "../include/model.h"
#include "../include/utils.h"
#include <mpi.h>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    MPI_Barrier(MPI_COMM_WORLD);
    // Iterate over training batches
    for (int j = 0; j < 100; j++) {
        for (uint32_t i = 0; i < data.train.num_batches; i++) {
            load_batch(&data, 1, i, batch_images, batch_labels);
            printf("Train batch %u: First label = %u: rank = %d\n", i, batch_labels[0], rank);
            // print_grayscale_image(batch_images, 28, 28);
            // fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
            forward(model, batch_images, batch_labels);
            backward(model);
            // Train your neural network here
        }
    }

    free(batch_images);
    free(batch_labels);
    free_mnist(&data);

    // MPI_Finalize();
    return 0;
}