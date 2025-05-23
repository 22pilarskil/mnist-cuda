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
    __shared__ float Asub[TILE_SIZE * TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + threadIdx.x < K)
            Asub[threadIdx.y * TILE_SIZE + threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < M)
            Bsub[threadIdx.y * TILE_SIZE + threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * M + col];
        else
            Bsub[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += Asub[threadIdx.y * TILE_SIZE + i] * Bsub[i * TILE_SIZE + threadIdx.x];
        __syncthreads();
    }

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
    #pragma omp parallel for
    for (int k = 0; k < in_dim * out_dim; k++) {
        int i = k / out_dim;
        int j = k % out_dim;
        weights_T[j * in_dim + i] = weights[i * out_dim + j];
    }
}

__global__ void cuda_transpose_kernel(float* weights, float* weights_T, int in_dim, int out_dim) {
    int i = blockIdx.x / out_dim;
    int j = blockIdx.x % out_dim;
    weights_T[j * in_dim + i] = weights[i * out_dim + j];
}

void cuda_transpose(float* weights, float* weights_T, int in_dim, int out_dim) {
    dim3 gridDim(in_dim * out_dim);
    cuda_transpose_kernel<<<gridDim, 1>>>(weights, weights_T, in_dim, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void host_multiply(float* weights, int weights_size, float coeff) {
    #pragma omp parallel for
    for (int i = 0; i < weights_size; i++) {
        weights[i] *= coeff;
    }
}

__global__ void cuda_multiply_kernel(float* weights, int weights_size, float coeff) {
    weights[blockIdx.x] *= coeff;
}

void cuda_multiply(float* weights, int weights_size, float coeff) {
    dim3 gridDim(weights_size);
    cuda_multiply_kernel<<<gridDim, 1>>>(weights, weights_size, coeff);
    CUDA_CHECK(cudaDeviceSynchronize());
}



void print_grayscale_image(float* img, int width, int height, FILE* out) {
    const char* shades = " .:-=+*#%@";  // 10 shades from light to dark
    FILE* target = out ? out : stdout;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float pixel = img[y * width + x];
            int shade_idx = (int)(pixel * 10);  // Map 0.0–1.0 to 0–9
            if (shade_idx < 0) shade_idx = 0;
            if (shade_idx > 9) shade_idx = 9;
            fprintf(target, "%c", shades[shade_idx]);
        }
        fprintf(target, "\n");
    }
    fflush(target);
}


void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    fflush(stdout);
}

char* get_dir_name() {
    char* dir_name = (char*)malloc(64 * sizeof(char));
    if (dir_name == NULL) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }
    char suffix[64] = "";
    if (USE_CUDA) {
        strcat(suffix, "_cuda");
    }
    if (USE_MPI_WEIGHT_SHARING) {
        strcat(suffix, "_weightsharing");
    }
    if (USE_MPI_MODEL_PARALLELISM) {
        strcat(suffix, "_modelparallelism");
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

void write_results(int epoch, float avg_accuracy, float avg_loss, int rank, float time_spent) {
    char filename[128];
    char* dir_name = get_dir_name();
    snprintf(filename, sizeof(filename), "%s/%d.txt", dir_name, rank);
    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "Epoch %d: Avg Accuracy = %f, Avg Loss = %f, Avg Time = %f\n", epoch, avg_accuracy, avg_loss, time_spent);
    fclose(fp);
    free(dir_name);
}