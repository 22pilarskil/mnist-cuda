#include "../../include/model.h"
#include "../../include/macros.h"

typedef struct {
    int dim;
} InputBuffer;

Layer* initInputBuffer(int batch_size, int dim);
void inputBuffer_forward(Layer* layer, float* inputs, int batch_size);
void inputBuffer_update(Layer* layer, int batch_size);


