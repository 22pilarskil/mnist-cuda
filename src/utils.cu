#include "../include/model.h"
#include "../include/macros.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#define TILE_SIZE 32

void host_matrix_multiply(float* A, float* B, float* C, int N, int K, int M) {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C[i * M + j] = A[i * K + 0] * B[0 * M + j];
        }
    }
    
    for (int k = 1; k < K; k++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            float temp = A[i * K + k];
            for (int j = 0; j < M; j++) {
                C[(i * M + j)] += temp * B[(k * M + j)];
            }
        }
    }
}

__global__ void cuda_matrix_multiply_kernel(float *A, float *B, float *C, int N, int K, int M) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    // Corrected indices: row in A (N) and col in B (M)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + threadIdx.x < K)
            Asub[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < M)
            Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * M + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        __syncthreads();
    }

    // Write to C (N×M matrix)
    if (row < N && col < M)
        C[row * M + col] = sum;
}

void cuda_matrix_multiply(float* A, float* B, float* C, int N, int K, int M) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    cuda_matrix_multiply_kernel<<<gridDim, blockDim>>>(A, B, C, N, K, M);
    CUDA_CHECK(cudaDeviceSynchronize());
}


void host_transpose(float* weights, float* weights_T, int in_dim, int out_dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < in_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            weights_T[j * in_dim + i] = weights[i * out_dim + j];
        }
    }
}

void print_grayscale_image(float* img, int width, int height) {
    const char* shades = " .:-=+*#%@";  // 10 shades from light to dark
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float pixel = img[y * width + x];
            int shade_idx = (int)(pixel * 10);  // Map 0-255 to 0-9
            printf("%c", shades[shade_idx]);
        }
        printf("\n");
    }
    fflush(stdout);
}

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void host_multiply(float* weights, int weights_size, float coeff) {
    #pragma omp parallel for
    for (int i = 0; i < weights_size; i++) {
        weights[i] *= coeff;
    }
}

char* get_dir_name() {
    char* dir_name = (char*)malloc(64 * sizeof(char));
    if (dir_name == NULL) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }
    char suffix[64] = "";
    if (USE_CUDA) {
        strcat(suffix, "_withcuda");
    }
    if (USE_MPI) {
        strcat(suffix, "_withmpi");
    }
    snprintf(dir_name, 64, "results%s", suffix);
    return dir_name;
}

void make_results_dir() {
    struct stat st = {0};
    char* dir_name = get_dir_name();
    if (stat(dir_name, &st) == -1) {
        if (mkdir(dir_name, 0700) == -1) {
            perror("Error creating directory");
            exit(EXIT_FAILURE);
        } else {
            printf("Directory %s created successfully.\n", dir_name);
        }
    } else {
        printf("Directory %s already exists.\n", dir_name);
    }
    free(dir_name);

}

void write_results(int epoch, float avg_accuracy, float avg_loss, int rank) {
    char filename[128];
    char* dir_name = get_dir_name();
    snprintf(filename, sizeof(filename), "%s/%d.txt", dir_name, rank);
    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "Epoch %d: Avg Accuracy = %f, Avg Loss = %f\n", epoch, avg_accuracy, avg_loss);
    fclose(fp);
    free(dir_name);
}