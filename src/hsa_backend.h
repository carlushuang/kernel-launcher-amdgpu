#ifndef __HSA_BACKEND_H
#define __HSA_BACKEND_H
#include "backend.h"
#include "hsa.h"
#include "hsa_ext_amd.h"

class hsa_backend_helper;

class hsa_backend : public backend{
public:
    friend class hsa_backend_helper;
    hsa_backend();
    ~hsa_backend();
    virtual int init_backend();     // init platform, find device
    virtual int setup_dispatch();   // prepare excutable, dispatch param
    virtual int dispatch();         // do dispatch
    virtual int wait();             // wait for finish

    virtual const char * name() const { return "hsa_backend"; }

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

};

#endif
