#include "hsa_backend.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdio.h>

#define HSA_ENFORCE(msg, rtn) \
            if(rtn != HSA_STATUS_SUCCESS) {\
                const char * err; \
                hsa_status_string(rtn, &err); \
                std::cerr<<"ERROR:"<<msg<<", rtn:"<<rtn<<", "<<err<<std::endl;\
                return -1; \
            }

#define HSA_ENFORCE_PTR(msg, ptr) \
            if(!ptr) {\
                std::cerr<<"ERROR:"<<msg<<std::endl;\
                return -1; \
            }

static inline uint16_t aql_header(hsa_packet_type_t type) {
    uint16_t header = type << HSA_PACKET_HEADER_TYPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    return header;
}

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
        if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
            hsa_backend* b = static_cast<hsa_backend*>(data);
            b->cpu_agent_ = agent;
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
    cpu_agent_.handle = 0;
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

    // Create a queue in the kernel agent. The queue can hold 4 packets, and has no callback or service queue associated with it
    status = hsa_queue_create(agent_, queue_size_, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue_);
    HSA_ENFORCE("hsa_queue_create", status);

    status = hsa_signal_create(1, 0, NULL, &signal_);
    HSA_ENFORCE("hsa_signal_create", status);

    status = hsa_agent_iterate_regions(agent_, hsa_backend_helper::get_region_callback, this);
    HSA_ENFORCE("hsa_agent_iterate_regions", status);

    HSA_ENFORCE_PTR("Failed to find kernarg memory region", kernarg_region_.handle)

    return 0;
}

int hsa_backend::setup_dispatch(dispatch_param * param){
    hsa_dispatch_param * d_param = (hsa_dispatch_param*)param;
    // Request a packet ID from the queue. Since no packets have been enqueued yet, the expected ID is zero
    packet_index_ = hsa_queue_add_write_index_relaxed(queue_, 1);
    const uint32_t queue_mask = queue_->size - 1;
    aql_ = (hsa_kernel_dispatch_packet_t*) (hsa_kernel_dispatch_packet_t*)(queue_->base_address) + (packet_index_ & queue_mask);
    const size_t aql_header_size = 4;
    memset((uint8_t*)aql_ + aql_header_size, 0, sizeof(*aql_) - aql_header_size);
    // initialize_packet
    aql_->completion_signal = signal_;
    aql_->workgroup_size_x = 1;
    aql_->workgroup_size_y = 1;
    aql_->workgroup_size_z = 1;
    aql_->grid_size_x = 1;
    aql_->grid_size_y = 1;
    aql_->grid_size_z = 1;
    aql_->group_segment_size = 0;
    aql_->private_segment_size = 0;
    // executable
    if (0 != load_bin_from_file(d_param->code_file_name.c_str()))
        return -1;
    hsa_status_t status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                 NULL, &executable_);
    HSA_ENFORCE("hsa_executable_create", status);
    // Load code object
    status = hsa_executable_load_code_object(executable_, agent_, code_object_, NULL);
    HSA_ENFORCE("hsa_executable_load_code_object", status);
    // Freeze executable
    status = hsa_executable_freeze(executable_, NULL);
    HSA_ENFORCE("hsa_executable_freeze", status);

    // Get symbol handle
    hsa_executable_symbol_t kernel_symbol;
    status = hsa_executable_get_symbol(executable_, NULL, d_param->kernel_symbol.c_str(), agent_,
                                        0, &kernel_symbol);
    HSA_ENFORCE("hsa_executable_get_symbol", status);

    // Get code handle
    uint64_t code_handle;
    status = hsa_executable_symbol_get_info(kernel_symbol,
                                            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                            &code_handle);
    HSA_ENFORCE("hsa_executable_symbol_get_info", status);
    status = hsa_executable_symbol_get_info(kernel_symbol,
                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                    &group_static_size_);
    HSA_ENFORCE("hsa_executable_symbol_get_info", status);

    aql_->kernel_object = code_handle;

    // kernel args
    void *kernarg;
    status = hsa_memory_allocate(kernarg_region_, d_param->kernel_arg_size, &kernarg);
    HSA_ENFORCE("hsa_memory_allocate", status);
    aql_->kernarg_address = kernarg;
    size_t kernarg_offset = 0;
    for(auto & ptr : d_param->kernel_arg_list){
        feed_kernarg(ptr.get(), kernarg_offset);
    }
    aql_->workgroup_size_x = d_param->local_size[0];
    aql_->workgroup_size_y = d_param->local_size[1];
    aql_->workgroup_size_z = d_param->local_size[2];

    aql_->grid_size_x = d_param->global_size[0];
    aql_->grid_size_y = d_param->global_size[1];
    aql_->grid_size_z = d_param->global_size[2];
    return 0;
}

void * hsa_backend::alloc(size_t size, void * param){
    hsa_region_t * region = static_cast<hsa_region_t *>(param);
    void *p = nullptr;
    hsa_status_t status = hsa_memory_allocate(*region, size, (void **)&p);
    if (status != HSA_STATUS_SUCCESS){
        std::cerr<<"hsa_memory_allocate failed, "<< status <<std::endl;
        return nullptr;
    }
    return p;
}
kernarg * hsa_backend::alloc_kernarg(size_t size){
    // alloc system memory
    void* system_ptr = alloc(size, &this->system_region_);
    if (!system_ptr) { return nullptr; }
#if 1
    // alloc local memory
    // gpu local first
    if (gpu_local_region_.handle != 0) {
        void* local_ptr = alloc(size, &this->gpu_local_region_);
        if (!local_ptr) { this->free(system_ptr); return nullptr; }
        return new kernarg(size, local_ptr, system_ptr, this);
    }

    if (local_region_.handle != 0) {
        void* local_ptr = alloc(size, &this->local_region_);
        if (!local_ptr) { this->free(system_ptr); return nullptr; }
        return new kernarg(size, local_ptr, system_ptr, this);
    } else if (gpu_local_region_.handle != 0) {
        void* local_ptr = alloc(size, &this->gpu_local_region_);
        if (!local_ptr) { this->free(system_ptr); return nullptr; }
        return new kernarg(size, local_ptr, system_ptr, this);
    } else {
        return new kernarg(size, nullptr, system_ptr, this);
    }
#endif
}

kernarg * hsa_backend::alloc_kernarg_pod(size_t bytes){
    void* system_ptr = this->alloc(bytes, &this->system_region_);
    if (!system_ptr) { return nullptr; }

    kernarg * ka = new kernarg(bytes, nullptr,system_ptr, this);
    ka->set_pod(true);
    return ka;
}

void   hsa_backend::free(void * mem){
    if(mem)
        hsa_memory_free(mem);
}
int hsa_backend::load_bin_from_file(const char * file_name){
    std::ifstream inf(file_name, std::ios::binary | std::ios::ate);
    //HSA_ENFORCE_PTR("failed to load file", inf);
    if (!inf) {
        std::cerr << "Error: failed to load " << file_name << std::endl;
        return -1;
    }
    size_t size = std::string::size_type(inf.tellg());
    char *ptr = (char*) this->alloc(size, &this->system_region_);
    HSA_ENFORCE_PTR("failed to allocate memory for code object", ptr);

    inf.seekg(0, std::ios::beg);
    std::copy(std::istreambuf_iterator<char>(inf),
                std::istreambuf_iterator<char>(),
                ptr);

    hsa_status_t status = hsa_code_object_deserialize(ptr, size, NULL, &code_object_);
    HSA_ENFORCE("hsa_code_object_deserialize", status);

    return 0;
}
void hsa_backend::feed_kernarg(kernarg * ka, size_t & offset){
    if(ka->pod())
        // direct copy pod value to kernarg_address
        feed_kernarg_raw(ka->system_ptr(), ka->size(), ka->size(), offset);
    else
        feed_kernarg_raw(&ka->local_ptr(), sizeof(void*), sizeof(void*), offset);
}
void hsa_backend::feed_kernarg_raw(const void * ptr, size_t size, size_t align, size_t & offset){
    //assert((align & (align - 1)) == 0);
    offset = ((offset + align - 1) / align) * align;
    memcpy( (char*)aql_->kernarg_address + offset, ptr, size);
    offset += size;
}
int hsa_backend::copy_to_local(kernarg * ka){
    hsa_status_t status;
    if(!ka->local_ptr() && !ka->system_ptr()){
        std::cerr<<"ERROR: this kernarg can not copy to local, with system_ptr:"
            <<ka->system_ptr()<<", local_ptr:"<<ka->local_ptr()<<std::endl;
        return -1;
    }
    status = hsa_memory_copy(ka->local_ptr(), ka->system_ptr(), ka->size());
    HSA_ENFORCE("hsa_memory_copy", status);
    return 0;
}
int hsa_backend::copy_from_local(kernarg * ka){
    hsa_status_t status;
    if(!ka->local_ptr() && !ka->system_ptr()){
        std::cerr<<"ERROR: this kernarg can not copy from local, with system_ptr:"
            <<ka->system_ptr()<<", local_ptr:"<<ka->local_ptr()<<std::endl;
        return -1;
    }
    // ?
    //status = hsa_memory_assign_agent(ka->local_ptr(), cpu_agent_, HSA_ACCESS_PERMISSION_RW);
    //HSA_ENFORCE("hsa_memory_assign_agent", status);
    status = hsa_memory_copy(ka->system_ptr(), ka->local_ptr(), ka->size());
    HSA_ENFORCE("hsa_memory_copy", status);
    return 0;
}
int hsa_backend::dispatch(){
    uint16_t header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    // total dimension
    uint16_t dim = 1;
    if (aql_->grid_size_y > 1)
        dim = 2;
    if (aql_->grid_size_z > 1)
        dim = 3;
    aql_->group_segment_size = group_static_size_;
    uint16_t setup = dim << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    uint32_t header32 = header | (setup << 16);

    __atomic_store_n((uint32_t*)aql_, header32, __ATOMIC_RELEASE);

    hsa_signal_store_relaxed(queue_->doorbell_signal, packet_index_);

    return 0;
}
int hsa_backend::wait(){
    hsa_signal_value_t result;
    result = hsa_signal_wait_acquire(signal_,
        HSA_SIGNAL_CONDITION_EQ, 0, ~0ULL, HSA_WAIT_STATE_ACTIVE);
    return 0;
}
