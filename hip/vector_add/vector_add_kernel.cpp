#include "hip/hip_runtime.h"

extern "C" __global__ void vector_add(float * in, float * out, int num){
    for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<num; i+= blockDim.x*gridDim.x){
        out[i] += in[i];
    }
}
