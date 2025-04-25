#pragma once

#include "../include/base_types.h"
#include "../include/layers/input_buffer.h"
#include "../include/layers/dense.h"
#include "../include/layers/leaky_relu.h"
#include "../include/layers/softmax.h"
#include "../include/layers/sigmoid.h"
#include "../include/layers/mpi_recv_buffer.h"
#include "../include/layers/mpi_send_buffer.h"
#include "../include/loss/cross_entropy.h"
#include <stdint.h>
#include <stdio.h>

float* forward(Model* model, float* inputs, uint8_t* targets);
void backward(Model* model);
void update(Model* model);
