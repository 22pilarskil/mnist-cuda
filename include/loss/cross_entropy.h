#include "../../include/model.h"

Loss* initCrossEntropyLoss(int batch_size, int dim, float* inputs);
void cross_entropy_forward(Loss* loss, int batch_size, uint8_t* targets);
void host_cross_entropy(Loss* loss, int batch_size, uint8_t* targets);