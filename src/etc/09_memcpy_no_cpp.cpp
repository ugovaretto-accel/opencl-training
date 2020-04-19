//Memcopy example w/ bandwidth tests.
//Author: Ugo Varetto
//Note: page-locked memory transfers might not work properly on systems
//sharing the same memory for both host and device (e.g. CPU)

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <ctime>
#include <cstring>
#include <vector>

#include "clutil.h"

typedef double real_t;

typedef std::vector< real_t > ByteArray;

using namespace std;
//------------------------------------------------------------------------------
double time_diff_ms(const timespec& start, const timespec& end) {
    return end.tv_sec * 1E3 +  end.tv_nsec / double(1E6)
           - (start.tv_sec * 1E3 + start.tv_nsec / double(1E6));  
}


//------------------------------------------------------------------------------
double copy_host_to_device_page_locked(const ByteArray& data,
                                       const CLEnv &clenv) {

    cl_int status = 0;
    //have OpenCL allocate a page-locked buffer with CL_MEM_ALLOC_HOST_PTR
    cl_mem buffer = clCreateBuffer(clenv.context,
                                   CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(real_t) * data.size(), 0, &status);

    check_cl_error(status, "clCreateBuffer");
    cl_event mapEvent;
    cl_event unmapEvent;
    // timespec tStart = {0,  0};
    // timespec tEnd = {0, 0};
    //clock_gettime(CLOCK_MONOTONIC, &tStart);
    //map buffer to host memory: this might trigger a device-host transfer;
    //in this case since no host data was transfered to the CL buffer, no
    //device --> host transfer should occur
    clFinish(clenv.commandQueue);
    void *hostPtr = clEnqueueMapBuffer(
        clenv.commandQueue,
        buffer,
        CL_TRUE,
        CL_MAP_WRITE,
        0,
        sizeof(real_t) * data.size(),
        0,
        0,
        0,
        &status);
    //std::vector<uint8_t> b(size);
    check_cl_error(status, "clEnqueueMapBuffer");
    check_cl_error(clWaitForEvents(1, &mapEvent), "clWaitForEvent");
    cl_ulong start = 0;
    size_t retSize = 0;
    status = clGetEventProfilingInfo(unmapEvent,
                                     CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong),
                                     &start,
                                     &retSize);
    cl_ulong end = 0;
    status = clGetEventProfilingInfo(unmapEvent,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong),
                                     &end,
                                     &retSize);                                
    
    //memcpy(hostPtr, &b[0], b.size());
    //the unmap function is what triggers the actual host --> device memory
    //transfer to keep host and device memory in sync; note that if both
    //host and device do use the same memory space the timings might be
    //meaningless and represent only the latency of the C function call itself
    status = clEnqueueUnmapMemObject(clenv.commandQueue,
                                     buffer,
                                     hostPtr,
                                     1,
                                     &mapEvent,
                                     &unmapEvent);
    check_cl_error(status, "clEnqueueUnmapMemObject");
    return double(end - start);


    // status = clFinish(clenv.commandQueue);
    // check_cl_error(status, "clFinish");
    // status = clWaitForEvents(1, &unmapEvent);
    // clock_gettime(CLOCK_MONOTONIC, &tEnd);
    // check_cl_error(status, "clWaitForEvents");
    // return time_diff_ms(tStart, tEnd);
    // status = clGetEventProfilingInfo(unmapEvent,
    //                                  CL_PROFILING_COMMAND_START,
    //                                  sizeof(cl_ulong),
    //                                  &start,
    //                                  &retSize);
    // check_cl_error(status, "clGetEventProfilingInfo");
    // cl_ulong end = 0;
    // status = clGetEventProfilingInfo(unmapEvent,
    //                                  CL_PROFILING_COMMAND_END,
    //                                  sizeof(cl_ulong),
    //                                  &end,
    //                                  &retSize);

    // return double(end - start);
}

//------------------------------------------------------------------------------
// double copy_device_to_host_page_locked(const CLEnv& clenv,
//                                        const cl::Context &context,
//                                        cl::CommandQueue &queue) {

//     cl_int status = -1;
//     //have OpenCL allocate a page-locked buffer with CL_MEM_ALLOC_HOST_PTR
//     cl_mem buffer = clCreateBuffer(clenv.context,
//                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
//                                    size, 0, &status);
//     cl_event profileEvent;
//     //map buffer to host memory: this might trigger a device-host transfer;
//     //in this case since no host data was transfered to the CL buffer, no
//     //device --> host transfer should occur
//     void *hostPtr = clEnqueueMapBuffer(
//         clenv.commandQueue,
//         buffer,
//         CL_TRUE,
//         CL_MAP_WRITE,
//         0,
//         size,
//         0,
//         0,
//         0,
//         &status
//     );

//     cl_int status = -1;
//     //have OpenCL allocate a page-locked buffer with CL_MEM_ALLOC_HOST_PTR
//     cl_mem buffer = clCreateBuffer(clenv.context,
//                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
//                                    size, 0, &status);
//     cl::Buffer buffer(context,
//                       CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
//                       data.size(), 0);
//     cl::Event profileEvent;
//     queue.finish();

//     //actual data transfer operations do happen at map/unmap time

//     //in order to force an actual device --> host transfer we need to somehow
//     //"touch" the device memory by first enqueueing a host --> device
//     //transfer through unmap; not doing this might not trigger a
//     //device --> host transfer when mapping memory

//     //dummy host --> device transfer - not timed
//     void *hostPtr = queue.enqueueMapBuffer(buffer,
//                                            CL_TRUE,
//                                            CL_MAP_WRITE,
//                                            0,
//                                            data.size(),
//                                            0,
//                                            0);
//     queue.enqueueUnmapMemObject(buffer, hostPtr, 0, 0);
//     //device to host transfer
//     hostPtr = queue.enqueueMapBuffer(buffer,
//                                      CL_TRUE,
//                                      CL_MAP_WRITE,
//                                      0,
//                                      data.size(),
//                                      0,
//                                      &profileEvent);
//     queue.enqueueUnmapMemObject(buffer, hostPtr, 0, 0);
//     if (hostPtr == 0)
//         throw std::runtime_error("ERROR - NULL host pointer");

//     // Configure event processing
//     const cl_ulong start = profileEvent
//                                .getProfilingInfo<CL_PROFILING_COMMAND_START>();
//     const cl_ulong end = profileEvent
//                              .getProfilingInfo<CL_PROFILING_COMMAND_END>();
//     return double(end - start) / 1E6;
// }

//------------------------------------------------------------------------------
double GBs(size_t sizeInBytes, double timeInSeconds) {
    return (double(sizeInBytes) / 0x40000000) / timeInSeconds;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 5) {
        std::cout << "usage: " << argv[0]
                  << " <platform name>"
                     " <device type: default | cpu | gpu | acc>"
                     " <device num>"
                     " <size>"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    const int platformID = atoi(argv[1]);
    cl_device_type deviceType;
    const std::string dt(argv[2]);
    if (dt == "default")
        deviceType = CL_DEVICE_TYPE_DEFAULT;
    else if (dt == "cpu")
        deviceType = CL_DEVICE_TYPE_CPU;
    else if (dt == "gpu")
        deviceType = CL_DEVICE_TYPE_GPU;
    else if (dt == "acc")
        deviceType = CL_DEVICE_TYPE_ACCELERATOR;
    else {
        std::cerr << "ERROR - unrecognized device type " << dt << std::endl;
        exit(EXIT_FAILURE);
    }
    const int deviceID = atoi(argv[3]);
    const size_t SIZE = atoll(argv[4]);

    CLEnv clenv = create_clenv(argv[1], argv[2], atoi(argv[3]), true);

    cl_int status;

    ByteArray data(SIZE);
    
    const double h2dpl = copy_host_to_device_page_locked(data, clenv);
    //const double d2hpl = copy_device_to_host_page_locked(data, context, queue);

    std::cout
        << "  host to device - page locked: " << h2dpl << std::endl;
    //<< "  device to host - page locked: " << d2hpl << std::endl;
    std::cout
        << "  host to device - page locked: " << GBs(SIZE, h2dpl / 1E3)
        << std::endl;
    //<< "  device to host - page locked: " << GBs(SIZE, d2hpl / 1E3)
    //<< std::endl;

    return 0;
}