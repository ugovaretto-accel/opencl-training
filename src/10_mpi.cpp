//OpenCL/MPI example
//Author: Ugo Varetto 

//1) init data and copy to device; OpenCL kernels initialize data
//   with the MPI process id
//2) exchange data between nodes: copy data in/out of OpenCL buffers
//   through map/unmap
//3) execute OpenCL kernel: local data are incremented with the values
//   received from the other node
//4) copy data to host and validate: all values must be equal to
//   task id 0 + task id 1 = 1


// compilation:                                                                     
// mpicxx ../../../scratch/mpi-pingpong-gpu.cpp \
//        -I <path to OpenCL include dir> \
//        -L <path to OpenCL lib dir> \
//        -lOpenCL -o 10_mpi
// execution(MVAPICH2): 
// mpiexec.hydra -n 2 -ppn 1 ./10_mpi 0 default 0 128

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <mpi.h> // <-!
#include "cl.hpp"

typedef double real_t; 

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 5) {
        std::cout << "usage: " << argv[0]
                << " <platform id(0, 1, ...)>"
                   " <device type: default | cpu | gpu | acc>"
                   " <device id(0, 1, ...)>"
                   " <number of double prec. elements>\n";

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
    const size_t BYTE_SIZE = SIZE * sizeof(real_t);
    // init MPI environment
    MPI_Init(&argc, &argv);
    int task = -1;
   
    MPI_Comm_rank(MPI_COMM_WORLD, &task);
    try {
       
        //OpenCL init
        cl::Platform::get(&platforms);
        if(platforms.size() <= platformID) {
            std::cerr << "Platform id " << platformID << " is not available\n";
            exit(EXIT_FAILURE);
        }
   
        platforms[platformID].getDevices(deviceType, &devices);
        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[deviceID],
                               CL_QUEUE_PROFILING_ENABLE);

        std::vector< real_t > data(SIZE, -1);
        //device buffer #1: holds local data
        cl::Buffer devData(context,
                            CL_MEM_READ_WRITE 
                            | CL_MEM_ALLOC_HOST_PTR //<-- page locked memory
                            | CL_MEM_COPY_HOST_PTR, //<-- copy data from 'data'
                            BYTE_SIZE,
                            const_cast< double* >(&data[0]));
        //device buffer #2: holds data received from other node
        cl::Buffer devRecvData(context,
                            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            BYTE_SIZE);
        //process data on the GPU(set array elements to local MPI id)  
        const char CLCODE_INIT[] =
            "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
            "typedef double real_t;\n"
            "__kernel void arrayset(__global real_t* outputArray,\n"
            "                       real_t value) {\n"
            "//get global thread id for dimension 0\n"
            "const int id = get_global_id(0);\n"
            "outputArray[id] = value;\n" 
            "}";
    
        cl::Program::Sources initSource(1, 
                                        std::make_pair(CLCODE_INIT,
                                                       sizeof(CLCODE_INIT)));
        cl::Program initProgram(context, initSource);
        initProgram.build(devices);
        cl::Kernel initKernel(initProgram, "arrayset");        
        initKernel.setArg(0, devData);
        initKernel.setArg(1, real_t(task));
       
        queue.enqueueNDRangeKernel(initKernel,
                                 cl::NDRange(0),
                                 cl::NDRange(SIZE),
                                 cl::NDRange(1));

        //perform data exchange:
        //1) map device buffers to host memory
        void* sendHostPtr = queue.enqueueMapBuffer(devData,
                                               CL_FALSE,
                                               CL_MAP_READ,
                                               0,
                                               BYTE_SIZE);

        if(sendHostPtr == 0) throw std::runtime_error("NULL mapped ptr");
    
        void* recvHostPtr = queue.enqueueMapBuffer(devRecvData,
                                               CL_FALSE,
                                               CL_MAP_WRITE,
                                               0,
                                               BYTE_SIZE);
       
        if(recvHostPtr == 0) throw std::runtime_error("NULL mapped ptr");
        queue.finish();

        //2) copy data to from remote process
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

        MPI_Status status;
        if(task == 0) {
            MPI_Isend(sendHostPtr, SIZE, MPI_DOUBLE, dest,
                      tag0to1, MPI_COMM_WORLD, &send_req);
            MPI_Irecv(recvHostPtr, SIZE, MPI_DOUBLE, source,
                      tag1to0, MPI_COMM_WORLD, &recv_req);
        } else {
            MPI_Isend(sendHostPtr, SIZE, MPI_DOUBLE, dest,
                      tag1to0, MPI_COMM_WORLD, &send_req);
            MPI_Irecv(recvHostPtr, SIZE, MPI_DOUBLE, source,
                      tag0to1, MPI_COMM_WORLD, &recv_req);
        }
        //3) as soon as data is copied do unmap buffers, indirectlry
        //   triggering a host --> device copy
        MPI_Wait(&recv_req, &status);
        queue.enqueueUnmapMemObject(devRecvData, recvHostPtr);
        MPI_Wait(&send_req, &status);
        queue.enqueueUnmapMemObject(devData, sendHostPtr);

        //note that instead of having each process compile the code
        //you could e.g. send the size and content of the source buffer
        //to each process from root; or even send the precompiled code,
        //in this case all nodes of the clusted must be the same whereas
        //in the case of source code compilation hybrid systems are
        //automatically supported by OpenCL

        //process data on the GPU: increment local data array with value
        //received from other process
        const char CLCODE_COMPUTE[] =
            "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
            "typedef double real_t;\n"
            "__kernel void sum( __global const real_t* in,\n"
            "                   __global real_t* inout) {\n"
            "const int id = get_global_id(0);\n"
            "inout[id] += in[id];\n" 
            "}";
        cl::Program::Sources computeSource(1,
                                           std::make_pair(CLCODE_COMPUTE,
                                                          sizeof(CLCODE_COMPUTE)));
        cl::Program computeProgram(context, computeSource);
        computeProgram.build(devices);
        cl::Kernel computeKernel(computeProgram, "sum");        
        computeKernel.setArg(0, devRecvData);
        computeKernel.setArg(1, devData);

        queue.enqueueNDRangeKernel(computeKernel,
                                 cl::NDRange(0),
                                 cl::NDRange(SIZE),
                                 cl::NDRange(1));
        
        //map device data to host memory for validation and output
        real_t* computedDataHPtr = reinterpret_cast< real_t* >(
                                        queue.enqueueMapBuffer(devData,
                                               CL_FALSE,
                                               CL_MAP_READ,
                                               0,
                                               BYTE_SIZE));
        queue.finish();

        const int value = 1; // task id 0 + task id 1
        const std::vector< real_t > reference(SIZE, value);
        if(std::equal(computedDataHPtr, computedDataHPtr + SIZE,
                      reference.begin())) {
            std::cout << '[' << task << "]: PASSED" << std::endl;
        } else {
            std::cout << '[' << task << "]: FAILED" << std::endl;
        }
        //release mapped pointer
        queue.enqueueUnmapMemObject(devData, computedDataHPtr);
        //release MPI resources
        MPI_Finalize();
    } catch(cl::Error e) {
      std::cerr << e.what() << ": Error code " << e.err() << std::endl;
      MPI_Finalize();
      exit(EXIT_FAILURE);   
    }   
    return 0;
}



