#define USE_CUDA 1
#define USE_MPI 0
#define LR 0.01

#ifdef USE_CUDA
    #define COPY(dst, src, size) CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice))
#else
    #define COPY(dst, src, size) memcpy(dst, src, size)
#endif


#ifdef USE_CUDA
    #include <cuda_runtime.h>
    #define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s (%d)\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err), err); \
            fflush(stdout); \
            exit(1); \
        } \
    } while (0)
    #define MALLOC(ptr, size) CUDA_CHECK(cudaMallocManaged(ptr, size))
    #define FREE(ptr) CUDA_CHECK(cudaFree(ptr))
#else
    #include <stdlib.h>
    #define MALLOC(ptr, size) do { \
        *(ptr) = malloc(size); \
        if (*(ptr) == NULL) { \
            fprintf(stderr, "Memory allocation failed (%s:%d)\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
    #define FREE(ptr) free(ptr)
#endif