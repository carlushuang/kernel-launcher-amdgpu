#ifndef __HSA_BACKEND_H
#define __HSA_BACKEND_H
#include "backend.h"
#include "hsa.h"
#include "hsa_ext_amd.h"
#include <string>
#include <vector>
#include <memory>

class hsa_backend_helper;

class hsa_dispatch_param : public dispatch_param{
public:
    std::string             code_file_name;
    std::string             kernel_symbol;
    size_t                  kernel_arg_size;

    int local_size[3];
    int global_size[3];

    std::vector<std::unique_ptr<kernarg>>    kernel_arg_list;
    void emplace_kernarg(kernarg * ka){
        kernel_arg_list.emplace_back(std::unique_ptr<kernarg>(ka));
    }
};

class hsa_backend : public backend{
public:
    friend class hsa_backend_helper;
    hsa_backend();
    ~hsa_backend();
    virtual int init_backend();     // init platform, find device
    virtual int setup_dispatch(dispatch_param * param);   // prepare excutable, dispatch param
    virtual int dispatch();         // do dispatch
    virtual int wait();             // wait for finish

    virtual const char * name() const { return "hsa_backend"; }

    virtual void * alloc(size_t size, void * param);
    virtual kernarg * alloc_kernarg(size_t size);
    template <typename T>
    kernarg * alloc_kernarg_pod(T value){
        //
    }
    virtual void free(void * mem);

    virtual int load_bin_from_file(const char * file_name);

    //void alloc_kernarg_local(kernarg * ka);
    //void assign_kernarg(kernarg *ka);

    virtual int copy_to_local(kernarg * ka);
    virtual int copy_from_local(kernarg * ka);

    void feed_kernarg(kernarg * ka, size_t & offset);
    void feed_kernarg_raw(const void * ptr, size_t size, size_t align, size_t & offset);

private:
    hsa_agent_t agent_;
    hsa_agent_t cpu_agent_;
    uint32_t queue_size_;
    hsa_queue_t* queue_;
    hsa_signal_t signal_;

    // http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/memory.htm
    hsa_region_t system_region_;
    hsa_region_t kernarg_region_;
    hsa_region_t local_region_;
    hsa_region_t gpu_local_region_;

    // http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/architected_queuing_language_packets.htm
    hsa_kernel_dispatch_packet_t* aql_;
    uint64_t packet_index_;

    hsa_code_object_t code_object_;
    hsa_executable_t executable_;
    uint32_t group_static_size_;
};

#endif
