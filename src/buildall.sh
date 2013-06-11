#!/bin/sh
SRC=../src
CLSRC=../src/kernels
CLSDK=/opt/nvidia/cudatoolkit/5.0.35.102
CLLIB=/opt/cray/nvidia/default
g++ $SRC/01_device_query.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 01_device_query
g++ $SRC/02_create_context.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 02_create_context
g++ $SRC/03_kernel_load_and_exec.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 03_kernel_load_and_exec
g++ $SRC/04_matrix_multiply.cpp $SRC/clutil.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 04_matrix_multiply
g++ $SRC/05_dot_product.cpp $SRC/clutil.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 05_dot_product
g++ $SRC/06_matrix_multiply_timing.cpp $SRC/clutil.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 06_matrix_multiply_timing
g++ $SRC/07_convolution.cpp $SRC/clutil.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o 07_convolution
g++ $SRC/cl-compiler.cpp $SRC/clutil.cpp -I$CLSDK/include -L$CLLIB/lib64 -lOpenCL -o clcc
