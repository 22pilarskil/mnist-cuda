# Compilers
NVCC        = nvcc
MPICXX      = mpicxx -w

# Flags
NVCCFLAGS   = -arch=sm_90 -O3 -Xcudafe -w -Xcompiler -fopenmp
CFLAGS      = -fopenmp -w
LDFLAGS     = -lcudart -fopenmp -w

# Paths (adjust these based on your system)
CUDA_LIB_DIR = /usr/local/cuda/lib64
CUDA_INC_DIR = /usr/local/cuda/include

# Targets
TRAIN_TARGET = train
TEST_TARGET  = test

# Source files
COMMON_CUDA_SRCS = src/model.cu src/utils.cu $(wildcard src/layers/*.cu) $(wildcard src/loss/*.cu)
TRAIN_CUDA_SRCS  = src/train.cu
TEST_CUDA_SRCS   = src/test.cu
C_SRCS           = src/mnist_loader.c

# Object files
TRAIN_OBJS       = $(TRAIN_CUDA_SRCS:.cu=.o) $(COMMON_CUDA_SRCS:.cu=.o) $(C_SRCS:.c=.o)
TEST_OBJS        = $(TEST_CUDA_SRCS:.cu=.o) $(COMMON_CUDA_SRCS:.cu=.o) $(C_SRCS:.c=.o)

# Rules
all: $(TRAIN_TARGET) $(TEST_TARGET)

$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(MPICXX) $^ -o $@ $(LDFLAGS) -L$(CUDA_LIB_DIR)

$(TEST_TARGET): $(TEST_OBJS)
	$(MPICXX) $^ -o $@ $(LDFLAGS) -L$(CUDA_LIB_DIR)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I$(CUDA_INC_DIR)

%.o: %.c
	$(MPICXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TRAIN_TARGET) $(TEST_TARGET) $(TRAIN_OBJS) $(TEST_OBJS)

.PHONY: all clean