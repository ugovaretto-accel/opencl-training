#!/bin/bash
PLATFORM=$1
DIR=.
RUN=aprun
CLSRC=../src/kernels

echo $'\n=== 01_device_query ==='
$RUN $DIR/01_device_query
echo $'\n=== 02_create_context ==='
$RUN $DIR/02_create_context "$PLATFORM" default 0
echo $'\n=== 03_kernel_load_and_exec ==='
$RUN $DIR/03_kernel_load_and_exec "$PLATFORM" default 0 $CLSRC/03_kernel.cl arrayset
echo $'\n=== 04_matrix_multiply ==='
$RUN $DIR/04_matrix_multiply "$PLATFORM" default 0 $CLSRC/04_matrix_multiply.cl matmul
echo $'\n=== 04_matrix_multiply - block ==='
$RUN $DIR/04_matrix_multiply "$PLATFORM" default 0 $CLSRC/04_matrix_multiply.cl block_matmul
echo $'\n=== 05_dot_product ==='
$RUN $DIR/05_dot_product "$PLATFORM" default 0 $CLSRC/05_dot_product.cl dotprod
echo $'\n=== 06_matrix_multiply_timing ==='
$RUN $DIR/06_matrix_multiply_timing "$PLATFORM" default 0 $CLSRC/04_matrix_multiply.cl matmul 256 16
echo $'\n=== 06_matrix_multiply_timing - block ==='
$RUN $DIR/06_matrix_multiply_timing "$PLATFORM" default 0 $CLSRC/04_matrix_multiply.cl block_matmul 256 16
echo $'\n=== 07_convolution'
$RUN $DIR/07_convolution "$PLATFORM" default 0 $CLSRC/07_stencil.cl filter 258 16 std
echo $'\n=== 07_convolution - read from images write to buffer'
$RUN $DIR/07_convolution "$PLATFORM" default 0 $CLSRC/07_stencil.cl filter_image 258 16 image
echo $'\n=== 07_convolution - read from images write to image'
$RUN $DIR/07_convolution_image_write "$PLATFORM" default 0 $CLSRC/07_stencil.cl filter_image 258 16 image