# Compilers
NVCC        = nvcc
MPICXX      = mpicxx

# Flags
NVCCFLAGS   = -arch=sm_90 -O3
CFLAGS      = -Wall -fopenmp
LDFLAGS     = -lcudart

# Paths (adjust these based on your system)
CUDA_LIB_DIR = /usr/local/cuda/lib64
CUDA_INC_DIR = /usr/local/cuda/include

# Targets
TARGET      = train
CUDA_SRCS   = src/train.cu src/model.cu src/utils.cu $(wildcard src/layers/*.cu) $(wildcard src/loss/*.cu)
C_SRCS      = src/mnist_loader.c # Add the C source file
OBJS        = $(CUDA_SRCS:.cu=.o) $(C_SRCS:.c=.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) $^ -o $@ $(LDFLAGS) -L$(CUDA_LIB_DIR)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I$(CUDA_INC_DIR)

%.o: %.c
	$(MPICXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean