#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    int comm;
    int dim;
} MPIRecvBuffer;

Layer* initMPIRecvBuffer(int batch_size, int dim, int comm);
void mpi_recv_buffer_forward(Layer* layer, int batch_size);
void mpi_recv_buffer_backward(Layer* layer, int batch_size);
void mpi_recv_buffer_update(Layer* layer, int batch_size);


