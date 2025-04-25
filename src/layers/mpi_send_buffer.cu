#include "../../include/model.h"
#include "../../include/macros.h"
#include "../../include/utils.h"
#include <stdio.h>
#include <math.h>
#include <mpi.h>


Layer* initMPISendBuffer(int batch_size, int dim, int comm, float* inputs) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    MPISendBuffer* mpiSendBuffer = (MPISendBuffer*)malloc(sizeof(MPISendBuffer));
    mpiSendBuffer->dim = dim;
    mpiSendBuffer->comm = comm;

    MALLOC(&layer->outputs, batch_size * dim * sizeof(float));

    layer->forward = mpi_send_buffer_forward;
    layer->backward = mpi_send_buffer_backward;
    layer->weights_size = 0;
    MALLOC(&layer->downstream_grads, batch_size * dim * sizeof(float));
    layer->inputs = inputs;
    layer->layer_data = mpiSendBuffer;
    layer->type = LAYER_MPI_SEND_BUFFER;
    return layer;
}


void mpi_send_buffer_forward(Layer* layer, int batch_size) {
    MPISendBuffer* mpiSendBuffer = (MPISendBuffer*)layer->layer_data;  
    int dim = mpiSendBuffer->dim;
    int comm = mpiSendBuffer->comm;
    MPI_Send(layer->inputs, batch_size * dim, MPI_FLOAT, comm, 0, MPI_COMM_WORLD);
}

void mpi_send_buffer_backward(Layer* layer, int batch_size) {
    MPISendBuffer* mpiSendBuffer = (MPISendBuffer*)layer->layer_data;  
    int dim = mpiSendBuffer->dim;
    int comm = mpiSendBuffer->comm;
    MPI_Recv(layer->downstream_grads, batch_size * dim, MPI_FLOAT, comm, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
