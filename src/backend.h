#ifndef __BACKEND_H
#define __BACKEND_H

class backend {
public:
    backend() = default;
    ~backend() = default;
    //
    virtual int init_backend()=0;     // init platform, find device
    virtual int setup_dispatch()=0;   // prepare excutable, dispatch param
    virtual int dispatch()=0;         // do dispatch
    virtual int wait()=0;             // wait for finish

    virtual const char * name() const =0;
private:


};

#endif