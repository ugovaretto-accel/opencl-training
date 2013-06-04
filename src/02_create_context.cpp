#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

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
        status = clGetDeviceInfo(*i, CL_DEVICE_NAME,
                                 buf.size(), &buf[0], 0);
        check_cl_error(status, "clGetDeviceInfo");
        if(platformName == &buf[0]) {
            platformID = *i;
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
    int numDevices = 0; 
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
void print_cl_context_info(cl_context ctx) {
    std::cout << "OpenCL context info:" << std::endl;
    cl_int status;
    cl_uint refCount = 0;
    status = clGetContextInfo(ctx,
                              CL_CONTEXT_REFERENCE_COUNT,
                              sizeof(cl_uint),
                              &refCount);
    check_cl_error(status, "clGetContextInfo");
    std::cout << "  reference count:    " << refCount << std::endl;
    cl_device_id deviceID; //only a single device was selected
    status = clGetContextInfo(ctx,
                              CL_CONTEXT_DEVICES,
                              sizeof(cl_device_id),
                              &deviceID);
    check_cl_error(status, "clGetContextInfo");
    std::cout << "  device id reference: " << deviceID << std::endl;
    cl_context_properties ctxProps[3];
    status = clGetContextInfo(ctx,
                              CL_CONTEXT_PROPERTIES,
                              sizeof(cl_uint),
                              &ctxProps[0]);
    check_cl_error(status, "clGetContextInfo");
    cl_platform_id pid = cl_platform_id(ctxProps[1]);
    st::vector< char > buf(0x10000, 0);
    status = clGetPlatformInfo(pis, CL_PLATFORM_NAME, buf.size(), &buf[0], 0);
    check_cl_error(status, "clGetPlatformInfo");
    std::cout << "  platform:           " << buf << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 4) {
        std::cout << "usage: " << argv[0]
                  << "<platform name> <device type = default | cpu | gpu "
                     "| acc | all>  <device num>"
                  << std::endl;
        return 0; 
    }
    std::string platformName = argv[ 1 ];
    std::string deviceType = argv[2]; // use stringified enum
    int deviceNum = atoi(argv[3]);
    cl_context ctx = create_cl_context(platformName, deviceType, deviceNum)
    std::cout << "OpenCL context created" << std::endl;
    print_cl_context_info(ctx);
    check_cl_error(clReleaseContext(ctx), "clReleaseContext");
    std::cout << "OpenCL context released" << std::endl;
    return 0;
}