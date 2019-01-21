#!/bin/sh
CXX=/opt/rocm/bin/hipcc
SRC=vector_add.cpp
TARGET=vector_add

rm -rf $TARGET
$CXX -Wall -O3 $SRC -lm -o $TARGET
$CXX --genco  --targets gfx900,gfx906  vector_add_kernel.cpp -o vector-add-2.co
