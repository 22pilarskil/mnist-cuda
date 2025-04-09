#include "../include/model.h"
#include <stdint.h>
#include <stdio.h>


void host_matrix_multiply(float* A, float* B, float* C, int N, int K, int M) {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C[i * M + j] = A[i * M + 0] * B[0 * M + j];
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

void host_transpose(float* weights, float* weights_T, int in_dim, int out_dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < in_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            weights_T[j * out_dim + i] = weights[i * in_dim + j];
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
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}