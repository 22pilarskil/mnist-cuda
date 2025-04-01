#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/mnist_loader.h"

// Function to reverse bytes (MNIST files are big-endian)
static uint32_t reverse_uint32(uint32_t n) {
    return ((n >> 24) & 0xff) | ((n << 8) & 0xff0000) |
           ((n >> 8) & 0xff00) | ((n << 24) & 0xff000000);
}

// Load MNIST images from file and normalize to [0, 1]
static float* load_images(const char* filename, uint32_t* num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open image file");
        exit(1);
    }

    uint32_t magic, rows, cols;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(num_images, sizeof(uint32_t), 1, file);
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);

    magic = reverse_uint32(magic);
    *num_images = reverse_uint32(*num_images);
    rows = reverse_uint32(rows);
    cols = reverse_uint32(cols);

    if (magic != 2051 || rows != 28 || cols != 28) {
        fprintf(stderr, "Invalid image file format: %s\n", filename);
        fclose(file);
        exit(1);
    }

    // Allocate temporary buffer for raw uint8_t data
    uint8_t* raw_images = (uint8_t*)malloc(*num_images * IMAGE_SIZE);
    if (!raw_images) {
        perror("Failed to allocate memory for raw images");
        fclose(file);
        exit(1);
    }

    fread(raw_images, sizeof(uint8_t), *num_images * IMAGE_SIZE, file);
    fclose(file);

    // Allocate float array and normalize
    float* images = (float*)malloc(*num_images * IMAGE_SIZE * sizeof(float));
    if (!images) {
        perror("Failed to allocate memory for float images");
        free(raw_images);
        exit(1);
    }

    for (uint32_t i = 0; i < *num_images * IMAGE_SIZE; i++) {
        images[i] = raw_images[i] / 255.0f;  // Normalize to [0, 1]
    }

    free(raw_images);  // Free temporary buffer
    return images;
}

// Load MNIST labels from file (unchanged, still uint8_t)
static uint8_t* load_labels(const char* filename, uint32_t* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open label file");
        exit(1);
    }

    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(num_labels, sizeof(uint32_t), 1, file);

    magic = reverse_uint32(magic);
    *num_labels = reverse_uint32(*num_labels);

    if (magic != 2049) {
        fprintf(stderr, "Invalid label file format: %s\n", filename);
        fclose(file);
        exit(1);
    }

    uint8_t* labels = (uint8_t*)malloc(*num_labels);
    if (!labels) {
        perror("Failed to allocate memory for labels");
        fclose(file);
        exit(1);
    }

    fread(labels, sizeof(uint8_t), *num_labels, file);
    fclose(file);
    return labels;
}

// Load a single MNIST set (train or test)
static MNISTSet load_mnist_set(const char* image_file, const char* label_file,
                               uint32_t batch_size) {
    MNISTSet set;
    set.images = load_images(image_file, &set.num_images);
    set.labels = load_labels(label_file, &set.num_images);
    if (set.num_images == 0) {
        fprintf(stderr, "No data loaded from %s or %s\n", image_file, label_file);
        exit(1);
    }
    set.num_batches = (set.num_images + batch_size - 1) / batch_size;
    return set;
}

// Load both training and test MNIST datasets
MNISTData load_mnist(uint32_t batch_size, int rank) {
    if (batch_size == 0) {
        fprintf(stderr, "Batch size must be greater than 0\n");
        exit(1);
    }

    MNISTData data;
    data.batch_size = batch_size;

    char train_images_path[100];
    char train_labels_path[100];
    char test_images_path[100];
    char test_labels_path[100];

    sprintf(train_images_path, "chunks/chunk_%d/train-images-idx3-ubyte", rank);
    sprintf(train_labels_path, "chunks/chunk_%d/train-labels-idx1-ubyte", rank);
    sprintf(test_images_path, "chunks/chunk_%d/t10k-images-idx3-ubyte", rank);
    sprintf(test_labels_path, "chunks/chunk_%d/t10k-labels-idx1-ubyte", rank);

    data.train = load_mnist_set(train_images_path, train_labels_path, batch_size);
    data.test = load_mnist_set(test_images_path, test_labels_path, batch_size);

    printf("Loaded %u training images, %u batches\n", data.train.num_images, data.train.num_batches);
    printf("Loaded %u test images, %u batches\n", data.test.num_images, data.test.num_batches);

    return data;
}

// Free the MNIST dataset memory
void free_mnist(MNISTData* data) {
    free(data->train.images);
    free(data->train.labels);
    free(data->test.images);
    free(data->test.labels);
}

// Load a batch of images and labels from either train or test set
void load_batch(MNISTData* data, int is_train, uint32_t batch_num,
                float* batch_images, uint8_t* batch_labels) {
    MNISTSet* set = is_train ? &data->train : &data->test;
    uint32_t start = batch_num * data->batch_size;
    if (start >= set->num_images) {
        fprintf(stderr, "Batch number %u out of range for %s set (max %u)\n",
                batch_num, is_train ? "train" : "test", set->num_batches - 1);
        return;
    }

    uint32_t actual_size = data->batch_size;
    if (start + data->batch_size > set->num_images) {
        actual_size = set->num_images - start;
    }

    memcpy(batch_images, set->images + start * IMAGE_SIZE,
           actual_size * IMAGE_SIZE * sizeof(float));
    memcpy(batch_labels, set->labels + start,
           actual_size * sizeof(uint8_t));
}

// Example usage (optional, can be removed)
#ifdef MNIST_LOADER_MAIN
int main() {
    uint32_t batch_size = 64;
    MNISTData data = load_mnist(batch_size, 0);  // rank 0 for testing

    float* batch_images = (float*)malloc(batch_size * IMAGE_SIZE * sizeof(float));
    uint8_t* batch_labels = (uint8_t*)malloc(batch_size);
    if (!batch_images || !batch_labels) {
        perror("Failed to allocate batch memory");
        free_mnist(&data);
        exit(1);
    }

    // Load and print first training batch
    load_batch(&data, 1, 0, batch_images, batch_labels);
    printf("First training batch: Label = %u, First pixel = %.4f\n",
           batch_labels[0], batch_images[0]);

    // Load and print first test batch
    load_batch(&data, 0, 0, batch_images, batch_labels);
    printf("First test batch: Label = %u, First pixel = %.4f\n",
           batch_labels[0], batch_images[0]);

    free(batch_images);
    free(batch_labels);
    free_mnist(&data);
    return 0;
}
#endif