#include "hsa_backend.h"
#include <iostream>


#define HSA_ENFORCE(msg, rtn) \
            if(rtn != HSA_STATUS_SUCCESS) {\
                const char * err; \
                hsa_status_string(rtn, &err); \
                std::cerr<<"ERROR:"<<msg<<", rtn:"<<rtn<<", "<<err<<std::endl;\
                return -1; \
            }\

class hsa_backend_helper{
public:
    static hsa_status_t get_agent_callback(hsa_agent_t agent, void *data){
        if (!data)
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;

        hsa_device_type_t hsa_device_type;
        hsa_status_t hsa_error_code = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
        if (hsa_error_code != HSA_STATUS_SUCCESS)
            return hsa_error_code;

        if (hsa_device_type == HSA_DEVICE_TYPE_GPU) {
            hsa_backend* b = static_cast<hsa_backend*>(data);
            b->agent_ = agent;
        }

        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t get_region_callback(hsa_region_t region, void* data)
    {
        hsa_region_segment_t segment_id;
        hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);

        if (segment_id != HSA_REGION_SEGMENT_GLOBAL) {
            return HSA_STATUS_SUCCESS;
        }

        hsa_region_global_flag_t flags;
        bool host_accessible_region = false;
        hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
        hsa_region_get_info(region, (hsa_region_info_t)HSA_AMD_REGION_INFO_HOST_ACCESSIBLE, &host_accessible_region);

        hsa_backend* b = static_cast<hsa_backend*>(data);

        if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
            b->system_region_ = region;
        }

        if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
            if(host_accessible_region){
                b->local_region_ = region;
            }else{
                dispatch->SetGPULocalRegion(region);
                b->gpu_local_region_ = region;
            }
        }

        if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
            b->kernarg_region_ = region;
        }

        return HSA_STATUS_SUCCESS;
    }
};


hsa_backend::hsa_backend(){
    agent_.handle = 0;
    signal_.handle = 0;
    kernarg_region_.handle = 0;
    system_region_.handle = 0;
    local_region_.handle = 0;
    gpu_local_region_.handle = 0;
}
hsa_backend::~hsa_backend(){

}
//   http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/example_simple_dispatch.htm
int hsa_backend::init_backend(){
    hsa_status_t status;
    status = hsa_init();
    HSA_ENFORCE("hsa_init", status);

    // Find GPU
    status = hsa_iterate_agents(hsa_backend_helper::get_agent_callback, this);
    HSA_ENFORCE("hsa_iterate_agents", status);

    char agent_name[64];

    status = hsa_agent_get_info(agent_, HSA_AGENT_INFO_NAME, agent_name);
    HSA_ENFORCE("hsa_agent_get_info(HSA_AGENT_INFO_NAME)", status);

    std::cout << "Using agent: " << agent_name << std::endl;

    status = hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size_);
    HSA_ENFORCE("hsa_agent_get_info(HSA_AGENT_INFO_QUEUE_MAX_SIZE", status);

    status = hsa_queue_create(agent_, queue_size_, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue_);
    HSA_ENFORCE("hsa_queue_create", status);

    status = hsa_signal_create(1, 0, NULL, &signal_);
    HSA_ENFORCE("hsa_signal_create", status);

    status = hsa_agent_iterate_regions(agent_, get_region_callback, this);
    HSA_ENFORCE("hsa_agent_iterate_regions", status);

    if (!kernarg_region.handle) {
        std::cerr<<"Failed to find kernarg memory region"<<std::endl;
        return -1;
    }

    return 0;
}
int hsa_backend::setup_dispatch(){

}
int hsa_backend::dispatch(){

}
int hsa_backend::wait(){

}
