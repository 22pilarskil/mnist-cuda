#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include <stdio.h>
#include <math.h>
#include <mpi.h>


Layer* initMPIRecvBuffer(int batch_size, int dim, int comm, float* inputs) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    MPIRecvBuffer* mpiRecvBuffer = (MPIRecvBuffer*)malloc(sizeof(MPIRecvBuffer));
    mpiRecvBuffer->dim = dim;
    mpiRecvBuffer->comm = comm;

    MALLOC(&layer->outputs, batch_size * dim * sizeof(float));

    layer->forward = mpi_recv_buffer_forward;
    layer->backward = mpi_recv_buffer_backward;
    layer->weights_size = 0;
    MALLOC(&layer->downstream_grads, batch_size * dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = mpiRecvBuffer;
    layer->type = LAYER_MPI_RECV_BUFFER;
    return layer;
}


void mpi_recv_buffer_forward(Layer* layer, int batch_size) {
    MPIRecvBuffer* mpiRecvBuffer = (MPIRecvBuffer*)layer->layer_data;  
    int dim = mpiRecvBuffer->dim;
    int comm = mpiRecvBuffer->comm;
    MPI_Recv(layer->outputs, batch_size * dim, MPI_FLOAT, comm, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void mpi_recv_buffer_backward(Layer* layer, int batch_size) {
    MPIRecvBuffer* mpiRecvBuffer = (MPIRecvBuffer*)layer->layer_data;  
    int dim = mpiRecvBuffer->dim;
    int comm = mpiRecvBuffer->comm;
    MPI_Send(layer->upstream_grads, batch_size * dim, MPI_FLOAT, comm, 0, MPI_COMM_WORLD);
}
