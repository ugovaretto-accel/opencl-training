//Load and execute a simple kernel which sets each array element to a specific
//value; thread count is retrieved, saved and printed at the end
//if compiled with openmp support it also generate additional threads from
//openmp
//Author: Ugo Varetto
//Launch as e.g.:
// ./03 "AMD Accelerated Parallel Processing" default 0 \
//       src/kernels/03_kernel.cl arrayset

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#ifdef _OPENMP
#error "OPENMP!"
#include <omp.h>
#endif
//------------------------------------------------------------------------------
struct ThreadCountRecord {
    std::string msg;
    std::string psout;
    ThreadCountRecord(const std::string& message,
                      const std::string& info) : msg(message), psout(info) {}
};

struct EOLCount {
    int count;
    EOLCount() : count(0) {}
    void operator()(const char& c) {
        if(c == '\n') ++count;
    }
};


std::ostream& operator <<(std::ostream& os, const ThreadCountRecord& t) {
    int threadcount = 0;
    EOLCount lines = std::for_each(t.psout.begin(), t.psout.end(), EOLCount()); 
    os << t.msg << '\n' << "thread count: " << (lines.count - 1) 
       << '\n' << t.psout;
}


//------------------------------------------------------------------------------
std::string threadreport() {
    const pid_t pid = getpid();
    std::ostringstream oss;
    oss << "ps uH p " << pid;
    FILE* f = popen(oss.str().c_str(), "r");
    std::string psout;
    std::vector< char > buffer(0x10000);
    size_t count = fread(&buffer[0], sizeof(char), buffer.size(), f);
    while(count) {
            buffer[count] = '\0';
            psout += &buffer[0];
            count = fread(&buffer[0], sizeof(char), buffer.size(), f);
    }
    pclose(f);
    return psout;
}

std::vector< ThreadCountRecord > threadlog;

void log_thread_count(const char* msg) {
    threadlog.push_back(ThreadCountRecord(msg, threadreport()));
}

template < typename FI >
void print_thread_count(FI begin, FI end) {
    std::ostream_iterator< ThreadCountRecord > 
        tout(std::cout, "\n=====================\n");
    std::copy(begin, end, tout);
}


//------------------------------------------------------------------------------
void check_cl_error(cl_int status, const char* msg) {
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR " << status << " -- " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

//------------------------------------------------------------------------------
void CL_CALLBACK context_callback(const char * errInfo,
                                  const void * private_info,
                                  size_t cb,
                                  void * user_data) {
    std::cerr << "ERROR - " << errInfo << std::endl;
    exit(EXIT_FAILURE);
}

//------------------------------------------------------------------------------
// returns context associated with single device only,
// to make it support multiple devices, a list of
// <device type, device num> pairs is required
cl_context create_cl_context(const std::string& platformName,
                             const std::string& deviceTypeName,
                             int deviceNum) {
    cl_int status = 0;
    //1) get platfors and search for platform matching platformName
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, 0, &numPlatforms);
    check_cl_error(status, "clGetPlatformIDs");
    if(numPlatforms < 1) {
        std::cout << "No OpenCL platforms found" << std::endl;
        exit(EXIT_SUCCESS);
    }
    typedef std::vector< cl_platform_id > PlatformIDs;
    PlatformIDs platformIDs(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, &platformIDs[0], 0);
    check_cl_error(status, "clGetPlatformIDs");
    std::vector< char > buf(0x10000, char(0));
    cl_platform_id platformID;
    PlatformIDs::const_iterator pi = platformIDs.begin();
    for(; pi != platformIDs.end(); ++pi) {
        status = clGetPlatformInfo(*pi, CL_PLATFORM_NAME,
                                 buf.size(), &buf[0], 0);
        check_cl_error(status, "clGetPlatformInfo");
        if(platformName == &buf[0]) {
            platformID = *pi;
            break; 
        }
    } 
    if(pi == platformIDs.end()) {
        std::cerr << "ERROR - Couldn't find platform " 
                  << platformName << std::endl;
        exit(EXIT_FAILURE);
    }
    //2) get devices of deviceTypeName type and store their ids into
    //   an array then select device id at position deviceNum
    cl_device_type deviceType;
    if(deviceTypeName == "default") 
        deviceType = CL_DEVICE_TYPE_DEFAULT;
    else if(deviceTypeName == "cpu")
        deviceType = CL_DEVICE_TYPE_CPU;
    else if(deviceTypeName == "gpu")
        deviceType = CL_DEVICE_TYPE_GPU;
    else if(deviceTypeName == "acc")
        deviceType = CL_DEVICE_TYPE_ACCELERATOR; 
    else if(deviceTypeName == "all")
        deviceType = CL_DEVICE_TYPE_CPU;
    else {
        std::cerr << "ERROR - device type " << deviceTypeName << " unknown"
                  << std::endl;
        exit(EXIT_FAILURE);          
    }                      
    cl_uint numDevices = 0; 
    status = clGetDeviceIDs(platformID, deviceType, 0, 0, &numDevices);
    check_cl_error(status, "clGetDeviceIDs");
    if(numDevices < 1) {
        std::cerr << "ERROR - Cannot find device of type " 
                  << deviceTypeName << std::endl;
        exit(EXIT_FAILURE);          
    }
    typedef std::vector< cl_device_id > DeviceIDs;
    DeviceIDs deviceIDs(numDevices);
    status = clGetDeviceIDs(platformID, deviceType, numDevices,
                            &deviceIDs[0], 0);
    check_cl_error(status, "clGetDeviceIDs");
    if(deviceNum < 0 || deviceNum >= numDevices) {
        std::cerr << "ERROR - device number out of range: [0," 
                  << (numDevices - 1) << ']' << std::endl;
        exit(EXIT_FAILURE);
    }
    cl_device_id deviceID = deviceIDs[deviceNum]; 
    //3) create and return context
    cl_context_properties ctxProps[] = {
        CL_CONTEXT_PLATFORM,
        cl_context_properties(platformID),
        0
    };
    //only a single device supported
    cl_context ctx = clCreateContext(ctxProps, 1, &deviceID,
                                     &context_callback, 0, &status);
    check_cl_error(status, "clCreateContext");
    return ctx;
}


//------------------------------------------------------------------------------
//read text from file and return it as a string
std::string load_text(const char* filepath) {
    std::ifstream src(filepath);
    if(!src) {
        std::cerr << "ERROR - Cannot open file " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string(std::istreambuf_iterator<char>(src),
                       std::istreambuf_iterator<char>());	
}


//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 6) {
        std::cerr << "usage: " << argv[0]
                  << " <platform name> <device type = default | cpu | gpu "
                     "| acc | all>  <device num> <OpenCL source file path>"
                     " <kernel name>"
                  << std::endl;
        exit(EXIT_FAILURE);  
    }
    std::string platformName = argv[ 1 ];
    std::string deviceType = argv[2]; 
    int deviceNum = atoi(argv[3]);

    log_thread_count("\n\nstart");
#ifdef _OPENMP    
    {
        // const int CHUNKSIZE = 100;
        // const int N = 1000;    
        // int i, chunk;
        // float a[N], b[N], c[N];
        // for (i=0; i < N; i++)
        //     a[i] = b[i] = i * 1.0;
        // chunk = CHUNKSIZE;
        // printf("%d", omp_get_max_threads());
        #pragma omp parallel //shared(a,b,c,chunk) private(i)
        {
            if(omp_get_thread_num() == 0) {
                log_thread_count("OMP PARALLEL");
            }
        // printf("%d", omp_get_max_threads());
        // #pragma omp for schedule(dynamic,chunk) nowait
        // for (i=0; i < N; i++) {
        //     c[i] = a[i] + b[i];
        // }
        }  
        #pragma omp barrier
    }
#endif    

    //1)create context
    cl_context ctx = create_cl_context(platformName, deviceType, deviceNum);
    std::cout << "OpenCL context created" << std::endl;
    
    log_thread_count("CL context created");


    //2)load kernel source
    const std::string programSource = load_text(argv[4]);
    std::cout << "OpenCL source code loaded" << std::endl;
    const char* src = programSource.c_str();
    const size_t sourceLength = programSource.length();

    //3)build program and create kernel
    cl_int status;
    cl_program program = clCreateProgramWithSource(ctx, //context
                                                   1,   //number of strings
                                                   &src, //source
                                                   &sourceLength, // size 
                                                   &status);  // status 
    check_cl_error(status, "clCreateProgramWithSource");
    cl_device_id deviceID; //only a single device was selected
    // retrieve actual device id from context
    status = clGetContextInfo(ctx,
                              CL_CONTEXT_DEVICES,
                              sizeof(cl_device_id),
                              &deviceID, 0);
    check_cl_error(status, "clGetContextInfo");
    cl_int buildStatus = clBuildProgram(program, //program
                                        1, //number of devices
                                        &deviceID, //array of device ids
                                        0, //program options as passed on
                                           //the command line to regualar
                                           //compilers e.g. -DUSE_DOUBLE
                                        0, 0);
    //log output if any
    char buffer[0x10000] = "";
    size_t len = 0;
    status = clGetProgramBuildInfo(program,
                                   deviceID,
                                   CL_PROGRAM_BUILD_LOG,
                                   sizeof(buffer),
                                   buffer,
                                   &len);
    check_cl_error(status, "clBuildProgramInfo");
    if(len > 1) std::cout << "Build output: " << buffer << std::endl;
    check_cl_error(buildStatus, "clBuildProgram");

    std::cout << "Built OpenCL program" << std::endl;
    const char* kernelName = argv[5];
    cl_kernel kernel = clCreateKernel(program, kernelName, &status);
    check_cl_error(status, "clCreateKernel"); 
   
    //4)allocate output buffer on OpenCL device
    typedef float real_t;
    const size_t ARRAY_LENGTH = 16;
    const size_t ARRAY_BYTE_LENGTH = ARRAY_LENGTH * sizeof(real_t);
    cl_mem outputCLBuffer = clCreateBuffer(ctx,
                                           CL_MEM_WRITE_ONLY,
                                           ARRAY_BYTE_LENGTH,
                                           0,
                                           &status);
    check_cl_error(status, "clCreateBuffer");

    //5)create command queue
    cl_command_queue commands = clCreateCommandQueue(ctx, deviceID, 0, &status);
    check_cl_error(status, "clCreateCommandQueue");
   
    //6)set kernel parameters
    const real_t value = real_t(3.14);
    //first parameter: output array
    status = clSetKernelArg(kernel, //kernel
                            0,      //parameter id
                            sizeof(cl_mem), //size of parameter
                            &outputCLBuffer); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(0)");
    //second parameter: value to assign to each array element
    status = clSetKernelArg(kernel, //kernel
                            1,      //parameter id
                            sizeof(real_t), //size of parameter
                            &value); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(1)");

    //7)setup kernel launch configuration
    //total number of threads == number of array elements
    const size_t globalWorkSize[1] = {ARRAY_LENGTH};
    //number of per-workgroup local threads
    const size_t localWorkSize[1] = {1}; 

    //8)launch kernel
    status = clEnqueueNDRangeKernel(commands, //queue
                                    kernel, //kernel                                   
                                    1, //number of dimensions for work-items
                                    0, //global work offset
                                    globalWorkSize, //total number of threads
                                    localWorkSize, //threads per workgroup
                                    0, //number of events that need to
                                       //complete before kernel executed
                                    0, //list of events that need to complete
                                       //before kernel executed
                                    0); //event object identifying this
                                        //particular kernel execution instance

    check_cl_error(status, "clEnqueueNDRangeKernel");
    log_thread_count("kernel launched");
    std::cout << "Lunched OpenCL kernel - setting all array elements to "
              << value << std::endl;

    //9)read back and print results
    std::vector< real_t > hostArray(ARRAY_LENGTH, real_t(0));
    status = clEnqueueReadBuffer(commands,
                                 outputCLBuffer,
                                 CL_TRUE, //blocking read
                                 0, //offset
                                 ARRAY_BYTE_LENGTH, //byte size of data
                                 &hostArray[0], //destination buffer in host
                                                // memory
                                 0, //number of events that need to
                                    //complete before transfer executed
                                 0, //list of events that need to complete
                                    //before transfer executed
                                 0); //event identifying this specific operation
    check_cl_error(status, "clEnqueueReadBuffer");
    log_thread_count("device -> host transfer");
    std::cout << "Output array: " << std::endl;
    std::ostream_iterator<real_t> out_it(std::cout, " ");
    std::copy(hostArray.begin(), hostArray.end(), out_it);
    std::cout << std::endl;

    //10)release resources
    check_cl_error(clReleaseMemObject(outputCLBuffer), "clReleaseMemObject");
    check_cl_error(clReleaseCommandQueue(commands),"clReleaseCommandQueue");
    check_cl_error(clReleaseKernel(kernel), "clReleaseKernel");
    check_cl_error(clReleaseProgram(program), "clReleaseProgram");
    check_cl_error(clReleaseContext(ctx), "clReleaseContext");
    std::cout << "Released OpenCL resources" << std::endl;
    log_thread_count("released resources");

    print_thread_count(threadlog.begin(), threadlog.end());
    std::cout << std::endl;
    return 0;
}


