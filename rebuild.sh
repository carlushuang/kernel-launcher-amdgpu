#!/bin/sh

rm -rf build
mkdir build
cd build

cmake -DLLVM_DIR=/opt/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/lib/cmake/llvm \
    ../ || exit 1
make -j8 || exit 1