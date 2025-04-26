#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    int comm;
    int dim;
} MPISendBuffer;

Layer* initMPISendBuffer(int batch_size, int dim, int comm, float* inputs);
void mpi_send_buffer_forward(Layer* layer, int batch_size);
void mpi_send_buffer_backward(Layer* layer, int batch_size);
void mpi_send_buffer_update(Layer* layer, int batch_size);

