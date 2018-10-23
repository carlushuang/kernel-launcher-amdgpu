#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>

#define HIPCHECK(error)                                                                  \
    {                                                                                    \
        hipError_t localError = error;                                                   \
        if (localError != hipSuccess) {                                                  \
            printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__ );                             \
            printf("API returned error code.");                                          \
            abort();                                                                     \
        }                                                                                \
    }

__global__
inline void vec_add(float * in, float * out, int num){
    for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<num; i+= blockDim.x*gridDim.x){
        out[i] += in[i];
    }
}

inline void host_vec_add(float * in, float * out, int num){
    for(int i=0;i<num;i++){
        out[i] += in[i];
    }
}

#define MIN_DELTA 1e-5
inline void valid_vec(float * host, float * dev, int num){
    float delta;
    bool valid = true;
    for(int i=0;i<num;i++){
        delta = fabsf(host[i] - dev[i]);
        if(delta > MIN_DELTA){
            printf("-- host/dev diff %f at %d, with %f, %f each, min %f\n", delta, i, host[i], dev[i], MIN_DELTA);
            valid = false;
        }
    }
    if(valid)
        printf("-- host/dev all valid with min delta %f\n", MIN_DELTA);
}

#define RANDOM_MIN .0f
#define RANDOM_MAX 10.f
void gen_vec(float * vec, int len){
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(RANDOM_MIN, RANDOM_MAX);
    for(int i=0;i<len;i++){
        vec[i] = dist(e2);
    }
}

#define VEC_LEN 128
#define GROUP_SIZE 64
#define GRID_SIZE 1
int main(){
    const int vec_len = VEC_LEN;
    float * host_in, * host_out, * host_out_2;
    float * dev_in, * dev_out;

    host_in = new float[vec_len];
    host_out = new float[vec_len];
    host_out_2 = new float[vec_len];
    gen_vec(host_in, vec_len);
    gen_vec(host_out, vec_len);

    HIPCHECK(hipSetDevice(0));
    HIPCHECK(hipMalloc(&dev_in, sizeof(float)*vec_len));
    HIPCHECK(hipMalloc(&dev_out, sizeof(float)*vec_len));

    HIPCHECK(hipMemcpy(dev_in, host_in, sizeof(float)*vec_len, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dev_out, host_out, sizeof(float)*vec_len, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(vec_add, dim3(GRID_SIZE), dim3(GROUP_SIZE), 0, 0, dev_in, dev_out, vec_len);
    HIPCHECK(hipMemcpy(host_out_2, dev_out, sizeof(float)*vec_len, hipMemcpyDeviceToHost));
    host_vec_add(host_in, host_out, vec_len);

    valid_vec(host_out, host_out_2, vec_len);

    hipFree(dev_in);
    hipFree(dev_out);
    delete [] host_in;
    delete [] host_out;
    delete [] host_out_2;
    hipDeviceReset();
}