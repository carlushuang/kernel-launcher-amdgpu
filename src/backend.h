#ifndef __BACKEND_H
#define __BACKEND_H

#include <cstring>
class dispatch_param{
public:
    dispatch_param()=default;
    ~dispatch_param()=default;
};

class kernarg;
class backend {
public:
    backend() = default;
    ~backend() = default;
    //
    virtual int init_backend()=0;     // init platform, find device
    virtual int setup_dispatch(dispatch_param * param)=0;   // prepare excutable, dispatch param
    virtual int dispatch()=0;         // do dispatch
    virtual int wait()=0;             // wait for finish

    virtual void * alloc(size_t size, void * param)=0;
    virtual kernarg * alloc_kernarg(size_t size) = 0;
    virtual kernarg * alloc_kernarg_pod(size_t bytes) = 0;
    virtual void   free(void * mem)=0;

    virtual const char * name() const =0;
    virtual int copy_to_local(kernarg * ka)=0;
    virtual int copy_from_local(kernarg * ka)=0;
private:

};

class kernarg{
public:
    kernarg(backend * back=nullptr){
        host_ptr_ = nullptr;
        dev_ptr_ = nullptr;
        bytes_ = 0;
        back_ = back;
        pod_ = false;
    }
    kernarg(size_t bytes, void * dev, void * host, backend * back=nullptr):
        bytes_(bytes), dev_ptr_(dev), host_ptr_(host), back_(back), pod_(false)
    {}
    ~kernarg(){
        //if(pod_){
        //    back_->free(host_ptr_);
        //    return;
        //}
        if(host_ptr_ && back_)
            back_->free(host_ptr_);
        if(dev_ptr_ && back_)
            back_->free(dev_ptr_);
    }

    size_t size() const {return bytes_;}
    bool  pod() const {return pod_;}
    bool  set_pod(bool pod){pod_ = pod;}

    void to_local(){
        if(back_)
            back_->copy_to_local(this);
    }
    void from_local(){
        if(back_)
            back_->copy_from_local(this);
    }

    // host side retrieve data
    template<typename T>
    T* data() { return static_cast<T*>(host_ptr_); }
    template<typename T>
    const T* data() const { return static_cast<const T*>(host_ptr_); }

    // device side retrieve data    
    void * & local_ptr(){ return dev_ptr_; }
    void * & system_ptr(){ return host_ptr_; }
private:
    void *      host_ptr_;
    void *      dev_ptr_;
    size_t      bytes_;
    backend *   back_;
    bool        pod_;
};


#endif