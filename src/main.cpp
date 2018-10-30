#include "hsa_backend.h"

#include <random>
#include <math.h>
#include <iostream>
#include <string>
#include <stdio.h>

int asm_kernel(){
    int rtn;
    backend * engine = new hsa_backend();
    rtn = engine->init_backend();
    if(rtn) return -1;
    std::cout<<"engine init ok"<<std::endl;

    hsa_dispatch_param d_param;

    kernarg * out = engine->alloc_kernarg(16);
    d_param.emplace_kernarg(out);
    d_param.code_file_name = "kernel/asm-kernel.co";
    d_param.kernel_symbol = "hello_world";
    d_param.kernel_arg_size = 1*sizeof(void *);

    d_param.local_size[0] = 1;
    d_param.local_size[1] = 0;
    d_param.local_size[2] = 0;
    d_param.global_size[0] = 1;
    d_param.global_size[1] = 0;
    d_param.global_size[2] = 0;

    rtn = engine->setup_dispatch(&d_param);
    if(rtn) return -1;
    std::cout<<"setup_dispatch ok"<<std::endl;

    rtn = engine->dispatch();
    if(rtn) return -1;
    std::cout<<"dispatch ok"<<std::endl;

    rtn = engine->wait();
    if(rtn) return -1;
    std::cout<<"wait ok"<<std::endl;

    out->from_local();
    std::cout<<"out:"<<  *out->data<float>()<<std::endl;
    return 0;
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
#define VEC_LEN 1000
#define GROUP_SIZE 64
#define GRID_SIZE 12

int vector_add(){
    int rtn;
    backend * engine = new hsa_backend();
    rtn = engine->init_backend();
    if(rtn) return -1;
    std::cout<<"engine init ok"<<std::endl;

    hsa_dispatch_param d_param;

    kernarg * ka_in = engine->alloc_kernarg(sizeof(float)*VEC_LEN);
    kernarg * ka_out = engine->alloc_kernarg(sizeof(float)*VEC_LEN);
    kernarg * ka_num = engine->alloc_kernarg_pod(sizeof(int));

    const int vec_len = VEC_LEN;
    float * host_in, * host_out;
    float * dev_in, * dev_out;

    host_in = new float[vec_len];
    host_out = new float[vec_len];
    //host_out_2 = new float[vec_len];
    gen_vec(host_in, vec_len);
    gen_vec(host_out, vec_len);


    for(int i=0;i<vec_len;i++){
        ka_in->data<float>()[i] = host_in[i];
        ka_out->data<float>()[i] = host_out[i];
    }
    *(ka_num->data<int>()) = vec_len;

    ka_in->to_local();
    ka_out->to_local();
    // ka_num->to_local();
    d_param.emplace_kernarg(ka_in);
    d_param.emplace_kernarg(ka_out);
    d_param.emplace_kernarg(ka_num);
    d_param.code_file_name = "kernel/vector-add-2.co";
    d_param.kernel_symbol = "vector_add";
    d_param.kernel_arg_size = 2*sizeof(void *) + sizeof(int);   // should be 20
    d_param.local_size[0] = GROUP_SIZE;
    d_param.local_size[1] = 0;
    d_param.local_size[2] = 0;
    d_param.global_size[0] = GROUP_SIZE * GRID_SIZE;
    d_param.global_size[1] = 0;
    d_param.global_size[2] = 0;
    rtn = engine->setup_dispatch(&d_param);
    if(rtn) return -1;
    std::cout<<"setup_dispatch ok"<<std::endl;

    std::cout<<"in ptr:"<<ka_in->local_ptr()<<std::endl;
    std::cout<<"out ptr:"<<ka_out->local_ptr()<<std::endl;
    std::cout<<"num ptr:"<<ka_num->local_ptr()<<std::endl;

    rtn = engine->dispatch();
    if(rtn) return -1;
    std::cout<<"dispatch ok"<<std::endl;

    rtn = engine->wait();
    if(rtn) return -1;
    std::cout<<"wait ok"<<std::endl;

    ka_out->from_local();
    host_vec_add(host_in, host_out, vec_len);
    //std::cout<<"out:"<<  *out->data<float>()<<std::endl;
    valid_vec(host_out, ka_out->data<float>(), vec_len);

    delete [] host_in;
    delete [] host_out;
    return 0;
}

int main(int argc, char ** argv){

    //return asm_kernel();
    return vector_add();
}
