#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>

// MNIST constants
#define IMAGE_SIZE (28 * 28) // 784 pixels per image
#define TRAIN_SIZE 60000
#define TEST_SIZE  10000

// Structure to hold a single MNIST dataset (train or test)
typedef struct {
    float* images;      // Flattened images (num_images * 784 bytes)
    uint8_t* labels;      // Labels (num_images bytes)
    uint32_t num_images;  // Number of images/labels
    uint32_t num_batches; // Number of batches for a given batch size
} MNISTSet;

// Structure to hold both training and test datasets
typedef struct {
    MNISTSet train;       // Training dataset
    MNISTSet test;        // Test dataset
    uint32_t batch_size;  // Batch size used to compute num_batches
} MNISTData;

// Function declarations
MNISTData load_mnist(uint32_t batch_size, int rank, int size);
void free_mnist(MNISTData* data);
void load_batch(MNISTData* data, int is_train, uint32_t batch_num,
                float* batch_images, uint8_t* batch_labels);

#endif // MNIST_LOADER_H