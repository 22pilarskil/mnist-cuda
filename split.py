#!/usr/bin/env python3

import os
import shutil
import sys
import struct

# MNIST file paths
MNIST_DATA_DIR = "data"
TRAIN_IMAGES = os.path.join(MNIST_DATA_DIR, "train-images-idx3-ubyte/train-images-idx3-ubyte")
TRAIN_LABELS = os.path.join(MNIST_DATA_DIR, "train-labels-idx1-ubyte/train-labels-idx1-ubyte")
TEST_IMAGES = os.path.join(MNIST_DATA_DIR, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
TEST_LABELS = os.path.join(MNIST_DATA_DIR, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

# Output directory
CHUNKS_DIR = "chunks"

# MNIST constants
IMAGE_SIZE = 28 * 28  # 784 bytes per image
TRAIN_SIZE = 60000
TEST_SIZE = 10000

def read_mnist_file(file_path, is_image=True):
    """Read a MNIST .ubyte file and return header and data."""
    with open(file_path, "rb") as f:
        data = f.read()
    
    if is_image:
        # Images: magic (2051), num_images, rows, cols
        magic, num, rows, cols = struct.unpack(">IIII", data[:16])
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError(f"Invalid image file: {file_path}")
        header = data[:16]
        content = data[16:]
    else:
        # Labels: magic (2049), num_labels
        magic, num = struct.unpack(">II", data[:8])
        if magic != 2049:
            raise ValueError(f"Invalid label file: {file_path}")
        header = data[:8]
        content = data[8:]
    
    return header, content, num

def write_mnist_file(file_path, header, content, num_items):
    """Write a chunk of MNIST data to a new .ubyte file."""
    with open(file_path, "wb") as f:
        # Update the number of items in the header
        if len(header) == 16:  # Image file
            f.write(struct.pack(">IIII", 2051, num_items, 28, 28))
        else:  # Label file
            f.write(struct.pack(">II", 2049, num_items))
        f.write(content)

def split_mnist(n_chunks):
    """Split MNIST files into n_chunks directories."""
    # Remove chunks directory if it exists
    if os.path.exists(CHUNKS_DIR):
        shutil.rmtree(CHUNKS_DIR)
    os.makedirs(CHUNKS_DIR)

    # Read all MNIST files
    train_img_header, train_img_data, train_img_num = read_mnist_file(TRAIN_IMAGES, True)
    train_lbl_header, train_lbl_data, train_lbl_num = read_mnist_file(TRAIN_LABELS, False)
    test_img_header, test_img_data, test_img_num = read_mnist_file(TEST_IMAGES, True)
    test_lbl_header, test_lbl_data, test_lbl_num = read_mnist_file(TEST_LABELS, False)

    if train_img_num != TRAIN_SIZE or train_lbl_num != TRAIN_SIZE or \
       test_img_num != TEST_SIZE or test_lbl_num != TEST_SIZE:
        raise ValueError("Unexpected number of items in MNIST files")

    # Calculate chunk sizes
    train_chunk_size = TRAIN_SIZE // n_chunks
    test_chunk_size = TEST_SIZE // n_chunks

    # Split and write to chunk directories
    for i in range(n_chunks):
        chunk_dir = os.path.join(CHUNKS_DIR, f"chunk_{i}")
        os.makedirs(chunk_dir, exist_ok=True)

        # Training set
        train_start = i * train_chunk_size
        train_end = train_start + train_chunk_size if i < n_chunks - 1 else TRAIN_SIZE
        train_img_chunk = train_img_data[train_start * IMAGE_SIZE:(train_end * IMAGE_SIZE)]
        train_lbl_chunk = train_lbl_data[train_start:train_end]
        train_num = train_end - train_start

        write_mnist_file(os.path.join(chunk_dir, "train-images-idx3-ubyte"),
                        train_img_header, train_img_chunk, train_num)
        write_mnist_file(os.path.join(chunk_dir, "train-labels-idx1-ubyte"),
                        train_lbl_header, train_lbl_chunk, train_num)

        # Test set
        test_start = i * test_chunk_size
        test_end = test_start + test_chunk_size if i < n_chunks - 1 else TEST_SIZE
        test_img_chunk = test_img_data[test_start * IMAGE_SIZE:(test_end * IMAGE_SIZE)]
        test_lbl_chunk = test_lbl_data[test_start:test_end]
        test_num = test_end - test_start

        write_mnist_file(os.path.join(chunk_dir, "t10k-images-idx3-ubyte"),
                        test_img_header, test_img_chunk, test_num)
        write_mnist_file(os.path.join(chunk_dir, "t10k-labels-idx1-ubyte"),
                        test_lbl_header, test_lbl_chunk, test_num)

        print(f"Chunk {i}: {train_num} training samples, {test_num} test samples")

def main():
    if len(sys.argv) != 2:
        print("Usage: python split.py <number_of_chunks>")
        sys.exit(1)

    try:
        n_chunks = int(sys.argv[1])
        if n_chunks <= 0:
            raise ValueError("Number of chunks must be positive")
    except ValueError:
        print("Error: Please provide a valid positive integer for the number of chunks")
        sys.exit(1)

    split_mnist(n_chunks)
    print(f"Successfully split MNIST into {n_chunks} chunks in {CHUNKS_DIR}/")

if __name__ == "__main__":
    main()