#include "hsa_backend.h"

#include <iostream>
#include <string>

int main(int argc, char ** argv){
    int rtn;
    backend * engine = new hsa_backend();
    rtn = engine->init_backend();
    if(rtn) return -1;
    std::cout<<"engine init ok"<<std::endl;

    hsa_dispatch_param d_param;

    kernarg * out = engine->alloc_kernarg(1024);
    d_param.emplace_kernarg(out);
    d_param.code_file_name = "asm-kernel.co";
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