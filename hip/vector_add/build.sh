#!/bin/sh
CXX=/opt/rocm/bin/hipcc
SRC=vector_add.cpp
TARGET=vector_add

rm -rf $TARGET
$CXX -Wall -O3 $SRC -lm -o $TARGET