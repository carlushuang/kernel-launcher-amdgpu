#ifndef __HSA_BACKEND_H
#define __HSA_BACKEND_H
#include "backend.h"
#include "hsa.h"
#include "hsa_ext_amd.h"
#include <string>

class hsa_backend_helper;

class hsa_dispatch_param{
public:
    std::string     code_file_name;
    std::string     kernel_symbol;
    size_t          kernel_arg_size;

};

class hsa_backend : public backend{
public:
    friend class hsa_backend_helper;
    hsa_backend();
    ~hsa_backend();
    virtual int init_backend();     // init platform, find device
    virtual int setup_dispatch(void * param);   // prepare excutable, dispatch param
    virtual int dispatch();         // do dispatch
    virtual int wait();             // wait for finish

    virtual const char * name() const { return "hsa_backend"; }

    virtual void * alloc(size_t size, void * param);
    virtual void   free(void * mem);

    virtual int load_bin_from_file(const char * file_name);

private:
    hsa_agent_t agent_;
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
