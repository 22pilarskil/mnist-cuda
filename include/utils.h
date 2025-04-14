void host_matrix_multiply(float* A, float* B, float* C, int N, int K, int M);
void print_grayscale_image(float* img, int width, int height);
void print_matrix(float* matrix, int rows, int cols);
void host_transpose(float* weights, float* weights_T, int in_dim, int out_dim);
void host_multiply(float* weights, int weights_size, float coeff);
void get_dir_name();
void make_results_dir();
void write_results(int epoch, float avg_accuracy, float avg_loss, int rank);
void cuda_matrix_multiply(float* A, float* B, float* C, int N, int K, int M);