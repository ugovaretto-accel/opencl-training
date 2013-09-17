//Author: Ugo Varetto
//Dot product: example of parallel reduction; supports vector data types in
//kernel, in case vector data types such as double4 are used the
//CL_ELEMENT_SIZE constant must be initialized with the vector size e.g. 4
//for 4-element vectors: pass '4' as the last element on the command line.
//TO HAVE CORRECT RESULTS ALWAYS #define USE_DOUBLE
//
// using monotonic clock to compute time intervals: link with librt (-lrt)
// compilation:
// c++ 05_dot_product_vec_timing.cpp clutil.cpp -lOpenCL -lrt -DUSE_DOUBLE
// run without arguments to see a list of supported options
//
// sample execution with
// 256M (1024*1024*256) doubles,
// 64 thread group
// 4-component elements (double4)  
//
// ('aprun' on Cray) ./a.out "Intel(R) OpenCL" default 0 \
// ./src/kernels/05_dot_product_vec.cl dotprod 268435456 1024 8

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <sstream>
#include <limits>
#include <algorithm>
#include <numeric>
#include <ctime>

#include "clutil.h"

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

//------------------------------------------------------------------------------
double time_diff_ms(const timespec& start, const timespec& end) {
    return end.tv_sec * 1E3 +  end.tv_nsec / 1E6
           - (start.tv_sec * 1E3 + start.tv_nsec / 1E6);  
}

//------------------------------------------------------------------------------
std::vector< real_t > create_vector(int size) {
    std::vector< real_t > m(size);
    srand(time(0));
    for(std::vector<real_t>::iterator i = m.begin();
        i != m.end(); ++i) *i = rand() % 10; 
    return m;
}

//------------------------------------------------------------------------------
real_t host_dot_product(const std::vector< real_t >& v1,
                        const std::vector< real_t >& v2) {
   return std::inner_product(v1.begin(), v1.end(), v2.begin(), real_t(0));
}

//------------------------------------------------------------------------------
bool check_result(real_t v1, real_t v2, double eps) {
    if(double(std::fabs(v1 - v2)) > eps) return false;
    else return true; 
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {

    if(argc < 9) {
        std::cerr << "usage: " << argv[0]
                  << " <platform name> <device type = default | cpu | gpu "
                     "| acc | all>  <device num> <OpenCL source file path>"
                     " <kernel name> <size> <local size> <vec element width>"
                  << std::endl;
        exit(EXIT_FAILURE);   
    }
    const int SIZE = atoi(argv[argc - 3]); // number of elements
    const int CL_ELEMENT_SIZE = atoi(argv[argc - 1]); // number of per-element
                                                      // components
    const size_t BYTE_SIZE = SIZE * sizeof(real_t);
    const int BLOCK_SIZE = atoi(argv[argc - 2]); //local cache for reduction
                                                 //equal to local workgroup size
    const int REDUCED_SIZE = SIZE / BLOCK_SIZE;
    const int REDUCED_BYTE_SIZE = REDUCED_SIZE * sizeof(real_t);
    //setup text header that will be prefixed to opencl code
    std::ostringstream clheaderStream;
    clheaderStream << "#define BLOCK_SIZE " << BLOCK_SIZE      << '\n';
    clheaderStream << "#define VEC_WIDTH "  << CL_ELEMENT_SIZE << '\n';
#ifdef USE_DOUBLE    
    clheaderStream << "#define DOUBLE\n";
    const double EPS = 0.000000001;
#else
    const float EPS = 0.00001;
#endif
    const bool PROFILE_ENABLE_OPTION = true;    
    CLEnv clenv = create_clenv(argv[1], argv[2], atoi(argv[3]),
                               PROFILE_ENABLE_OPTION,
                               argv[4], argv[5], clheaderStream.str());
   
    cl_int status;
    //create input and output matrices
    std::vector<real_t> V1 = create_vector(SIZE);
    std::vector<real_t> V2 = create_vector(SIZE);
    real_t hostDot = std::numeric_limits< real_t >::quiet_NaN();
    real_t deviceDot = std::numeric_limits< real_t >::quiet_NaN();      
//ALLOCATE DATA AND COPY TO DEVICE    
    //allocate output buffer on OpenCL device
    //the partialReduction array contains a sequence of dot products
    //computed on sub-arrays of size BLOCK_SIZE
    cl_mem partialReduction = clCreateBuffer(clenv.context,
                                             CL_MEM_WRITE_ONLY,
                                             REDUCED_BYTE_SIZE,
                                             0,
                                             &status);
    check_cl_error(status, "clCreateBuffer");

    //allocate input buffers on OpenCL devices and copy data
    cl_mem devV1 = clCreateBuffer(clenv.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  BYTE_SIZE,
                                  &V1[0], //<-- copy data from V1
                                  &status);
    check_cl_error(status, "clCreateBuffer");                              
    cl_mem devV2 = clCreateBuffer(clenv.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  BYTE_SIZE,
                                  &V2[0], //<-- copy data from V2
                                  &status);
    check_cl_error(status, "clCreateBuffer");                              

    //set kernel parameters
    status = clSetKernelArg(clenv.kernel, //kernel
                            0,      //parameter id
                            sizeof(cl_mem), //size of parameter
                            &devV1); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(V1)");
    status = clSetKernelArg(clenv.kernel, //kernel
                            1,      //parameter id
                            sizeof(cl_mem), //size of parameter
                            &devV2); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(V2)");
    status = clSetKernelArg(clenv.kernel, //kernel
                            2,      //parameter id
                            sizeof(cl_mem), //size of parameter
                            &partialReduction); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(devOut)");
   

    //setup kernel launch configuration
    //total number of threads == number of array elements
    const size_t globalWorkSize[1] = {SIZE / CL_ELEMENT_SIZE};
    //number of per-workgroup local threads
    const size_t localWorkSize[1] = {BLOCK_SIZE}; 
//LAUNCH KERNEL
    // make sure all work on the OpenCL device is finished
    status = clFinish(clenv.commandQueue);
    check_cl_error(status, "clFinish");
    cl_event profilingEvent;
    timespec kernelStart = {0,  0};
    timespec kernelEnd = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &kernelStart);
    //launch kernel
    status = clEnqueueNDRangeKernel(clenv.commandQueue, //queue
                                    clenv.kernel, //kernel                                   
                                    1, //number of dimensions for work-items
                                    0, //global work offset
                                    globalWorkSize, //total number of threads
                                    localWorkSize, //threads per workgroup
                                    0, //number of events that need to
                                       //complete before kernel executed
                                    0, //list of events that need to complete
                                       //before kernel executed
                                    &profilingEvent); //event object associated
                                                      // with this particular
                                                      // kernel execution
                                                      // instance

    check_cl_error(status, "clEnqueueNDRangeKernel");
    status = clFinish(clenv.commandQueue); //ensure kernel execution is
    //terminated; used for timing purposes only; there is no need to enforce
    //termination when issuing a subsequent blocking data transfer operation
    check_cl_error(status, "clFinish");
    status = clWaitForEvents(1, &profilingEvent);
    clock_gettime(CLOCK_MONOTONIC, &kernelEnd);
    check_cl_error(status, "clWaitForEvents");
    //get_cl_time(profilingEvent);  //gives similar results to the following 
    const double kernelElapsedTime_ms = time_diff_ms(kernelStart, kernelEnd);
//READ DATA FROM DEVICE
    //read back and print results
    std::vector< real_t > partialDot(REDUCED_SIZE); 
    status = clEnqueueReadBuffer(clenv.commandQueue,
                                 partialReduction,
                                 CL_TRUE, //blocking read
                                 0, //offset
                                 REDUCED_BYTE_SIZE, //byte size of data
                                 &partialDot[0], //destination buffer in host
                                                 //memory
                                 0, //number of events that need to
                                    //complete before transfer executed
                                 0, //list of events that need to complete
                                    //before transfer executed
                                 &profilingEvent); //event identifying this
                                                   //specific operation
    check_cl_error(status, "clEnqueueReadBuffer");

    const double dataTransferTime_ms = get_cl_time(profilingEvent);

    timespec accStart = {0, 0};
    timespec accEnd   = {0, 0};

//FINAL REDUCTION ON HOST    
    clock_gettime(CLOCK_MONOTONIC, &accStart);
    deviceDot = std::accumulate(partialDot.begin(),
                                partialDot.end(), real_t(0));
    clock_gettime(CLOCK_MONOTONIC, &accEnd);
    const double accTime_ms = time_diff_ms(accStart, accEnd);

//COMPUTE DOT PRODUCT ON HOST
    timespec hostStart = {0, 0};
    timespec hostEnd = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &hostStart);
    hostDot = host_dot_product(V1, V2);
    clock_gettime(CLOCK_MONOTONIC, &hostEnd);
    const double host_time = time_diff_ms(hostStart, hostEnd);
//PRINT RESULTS
    std::cout << deviceDot << ' ' << hostDot << std::endl;

    if(check_result(hostDot, deviceDot, EPS)) {
        std::cout << "PASSED" << std::endl;
        std::cout << "kernel:         " << kernelElapsedTime_ms << "ms\n"
                  << "host reduction: " << accTime_ms << "ms\n"
                  << "total:          " << (kernelElapsedTime_ms + accTime_ms)  
                  << "ms" << std::endl;
        std::cout << "transfer:       " << dataTransferTime_ms 
                  << "ms\n" << std::endl;
        std::cout << "host:           " << host_time << "ms" << std::endl;
       
    } else {
        std::cout << "FAILED" << std::endl;
    }   

    check_cl_error(clReleaseMemObject(devV1), "clReleaseMemObject");
    check_cl_error(clReleaseMemObject(devV2), "clReleaseMemObject");
    check_cl_error(clReleaseMemObject(partialReduction), "clReleaseMemObject");
    release_clenv(clenv);
   
    return 0;
}