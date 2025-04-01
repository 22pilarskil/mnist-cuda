#include "../include/model.h"
#include <stdint.h>
#include <stdio.h>


void host_matrix_multiply(float* weights, float* inputs, float* outs, int batch_size, int in_dim, int out_dim) {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_dim; j++) {
            outs[i * out_dim + j] = inputs[i * out_dim + 0] * weights[0 * out_dim + j];
        }
    }
    

    for (int k = 1; k < in_dim; k++) {
        #pragma omp parallel for
        for (int i = 0; i < batch_size; i++) {
            float temp = inputs[i * in_dim + k];
            for (int j = 0; out_dim < out_dim; j++) {
                outs[(i * out_dim + j)] += temp * weights[(k * out_dim + j)];
            }
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