//OpenCL/MPI ping pong test
//Author: Ugo Varetto 

//1) init data and copy to device; OpenCL kernels initialize data
//   with the MPI process id
//2) exchange data between devices
//3) copy data from device to host and validate


// compilation:                                                                     
// mpicxx ../../../scratch/mpi-pingpong-gpu.cpp \
//        -I <path to OpenCL include dir> \
//        -L <path to OpenCL lib dir> \
//        -lOpenCL

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <algorithm>
#include <vector>
#include <mpi.h> // <-!
#include "../cl.hpp"

typedef double real_t; 

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 5) {
        std::cout << "usage: " << argv[0]
                << " <platform id(0, 1...)>"
                   " <device type: default | cpu | gpu | acc>"
                   " <OpenCL source>"
                   " <kernel function name>\n"
                   "Must use kernel that sets all the element of an array to"
                   " a value and accepts an output buffer and an int"
                << std::endl; 
        exit(EXIT_FAILURE);          
    }
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    const int platformID = atoi(argv[1]);
    cl_device_type deviceType;
    const std::string kernelName(argv[4]);
    const std::string dt = std::string(argv[2]);
    if(dt == "default") deviceType = CL_DEVICE_TYPE_DEFAULT;
    else if(dt == "cpu") deviceType = CL_DEVICE_TYPE_CPU;
    else if(dt == "gpu") deviceType = CL_DEVICE_TYPE_GPU;
    else if(dt == "acc") deviceType = CL_DEVICE_TYPE_ACCELERATOR;
    else {
      std::cerr << "ERROR - unrecognized device type " << dt << std::endl;
      exit(EXIT_FAILURE);
    } 
    const int deviceID = atoi(argv[3]);
    const size_t SIZE = atoll(argv[4]);
    // init MPI environment
    MPI_Init(&argc, &argv);
    int task = -1;
    const size_t SIZE = atoi(argv[1]);
    const size_t BYTE_SIZE = size * sizeof(double);
    MPI_Comm_rank(MPI_COMM_WORLD, &task);
    try {
       
        //OpenCL init
        cl::Platform::get(&platforms);
        if(platforms.size() <= platformID) {
            std::cerr << "Platform id " << platformID << " is not available\n";
        exit(EXIT_FAILURE);
   
        platforms[platformID].getDevices(deviceType, &devices);
        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[deviceID],
                               CL_QUEUE_PROFILING_ENABLE);

        std::vector< real_t > data(SIZE, -1);
        //device buffer #1
        cl::Buffer devData1(context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            BYTE_SIZE,
                            const_cast< void* >(&data[0]));
        //device buffer #2
        cl::Buffer devData2(context,
                            CL_MEM_READ | CL_MEM_ALLOC_HOST_PTR,
                            BYTE_SIZE,
                            0);
        //process data on the GPU(set to MPI id)  
        const char CLCODE_INIT[] =
            "typedef double real_t"
            "__kernel void arrayset(__global real_t* outputArray,\n"
            "                       real_t value) {\n"
            "//get global thread id for dimension 0\n"
            "const int id = get_global_id(0);\n"
            "outputArray[id] = value;\n" 
            "}";
    
        cl::Program::Sources initSource(1, std::make_pair(CLCODE_INIT,
                                        0)); // 0 means that the source strings
                                             // are NULL terminated
        cl::Program initProgram(context, initSource);
        initProgram.build(devices);
        cl::Kernel initKernel(initProgram, "arrayset");        
        initKernel.setArg(0, buffer);
        initKernel.setArg(1, real_t(task));
       
        queue.enqueueNDRangeKernel(kernel,
                                 cl::NDRange(0),
                                 cl::NDRange(SIZE),
                                 cl::NDRange(1),
                                 0, // wait events *
                                 0);        

        void* sendHostPtr = queue.enqueueMapBuffer(devData1,
                                               CL_FALSE,
                                               CL_MAP_READ,
                                               0,
                                               BYTE_SIZE,
                                               0,
                                               0);
    
        queue.finish();
        void* recvHostPtr = queue.enqueueMapBuffer(devData2,
                                               CL_TRUE,
                                               CL_MAP_WRITE,
                                               0,
                                               BYTE_SIZE,
                                               0,
                                               0);

    
        const int tag0to1 = 0x01;
        const int tag1to0 = 0x10;
        MPI_Request send_req;
        MPI_Request recv_req;
        int source = -1;
        int dest = -1;
        if(task == 0 ) {
            source = 1;
            dest   = 1;
        } else {
            source = 0;
            dest   = 0;
        }
        MPI_Isend(sendHostPtr, SIZE, MPI_DOUBLE, dest,
                  tag0to1, MPI_COMM_WORLD, &send_req);
        MPI_Irecv(recvHostPtr, SIZE, MPI_DOUBLE, source,
                  tag1to0, MPI_COMM_WORLD, &recv_req);
        MPI_Wait(&recv_req, &status);
        queue.enqueueUnmapMemObject(devData2, recvHostPtr, 0, 0);
        MPI_Wait(&send_req, &status);
        queue.enqueueUnmapMemObject(devData1, sendHostPtr, 0, 0);

        //note that instead of having each process compile the code
        //you could e.g. send the size and content of the source buffer
        //to each process from root; or even send the precompiled code,
        //in this case all nodes of the clusted must be the same whereas
        //in the case of source code compilation hybrid systems are
        //automatically supported by OpenCL

        //process data on the GPU(set to MPI id)  
        const char CLCODE_COMPUTE[] =
            "typedef double real_t"
            "__kernel void sum( __global const real_t* in,\n"
            "                   __global real_t* inout) {\n"
            "const int id = get_global_id(0);\n"
            "inout[id] += in[id];\n" 
            "}";
        cl::Program::Sources computeSource(1, std::make_pair(CLCODE_COMPUTE,
                                           0)); // 0 means that the source strings
                                             // are NULL terminated
        cl::Program computeProgram(context, computeSource);
        computeProgram.build(devices);
        cl::Kernel computeKernel(computeProgram, "sum");        
        computeKernel.setArg(0, devData2);
        computeKernel.setArg(1, devData1);

        real_t* computedDataHPtr = reinterpret_cast< real_t* >(
                                        queue.enqueueMapBuffer(devData2,
                                               CL_TRUE,
                                               CL_MAP_WRITE,
                                               0,
                                               BYTE_SIZE,
                                               0,
                                               0));
        const int value = 1; // task id 0 + task id 1
        if(std::equal(computedDataHPtr, computedDataHPtr + SIZE,
                      real_t(1)) {
            std::cout << '[' << taks << "]: PASSED" << std::endl;
        } else {
            std::cout << '[' << taks << "]: FAILED" << std::endl;
        }

        MPI_Finalize();
    } catch(cl::Error e) {
      std::cerr << e.what() << ": Error code " << e.err() << std::endl;
      MPI_Finalize();
      exit(EXIT_FAILURE);   
    }   
    return 0;
}



