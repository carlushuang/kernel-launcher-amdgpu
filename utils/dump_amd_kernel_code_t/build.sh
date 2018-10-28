#!/bin/sh
SRC=amd_kernel_code_t.cpp
TARGET=dump_amd_kernel_code_t
#CXXFLAGS=" -g -Wall"
CXXFLAGS=" -O2 -Wall "

rm -rf $TARGET
g++ $CXXFLAGS $SRC -o $TARGET