//Implementation of OpenCL utility functions
//Author: Ugo Varetto
#include "clutil.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorith>
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
cl_device_id get_device_id(cl_context ctx) {
    cl_device_id deviceID; 
    // retrieve actual device id from context
    status = clGetContextInfo(ctx,
                              CL_CONTEXT_DEVICES,
                              sizeof(cl_device_id),
                              &deviceID, 0);
    check_cl_error(status, "clGetContextInfo");
    return deviceID;
}

//------------------------------------------------------------------------------
void print_devices(cl_platform_id pid) {
    cl_uint numDevices = 0;
    cl_int status = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 
                                   0, 0, &numDevices );
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR - clGetDeviceIDs" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(numDevices < 1) return;
    
    typedef std::vector< cl_device_id > DeviceIds;
    DeviceIds devices(numDevices);
    status = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL,
                            devices.size(), &devices[ 0 ], 0 );
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR - clGetDeviceIDs" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector< char > buf(0x10000, char(0));
    cl_uint d;
    std::cout << "Number of devices: " << devices.size() << std::endl;
    int dev = 0;
    for(DeviceIds::const_iterator i = devices.begin();
        i != devices.end(); ++i, ++dev) {
        std::cout << "Device " << dev <<  std::endl;
        // device type
        cl_device_type dt = cl_device_type();
        status = clGetDeviceInfo(*i, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type), &dt, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_TYPE)" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Type: "; 
        if( dt & CL_DEVICE_TYPE_DEFAULT     ) std::cout << "Default ";
        if( dt & CL_DEVICE_TYPE_CPU         ) std::cout << "CPU ";
        if( dt & CL_DEVICE_TYPE_GPU         ) std::cout << "GPU ";
        if( dt & CL_DEVICE_TYPE_ACCELERATOR ) std::cout << "Accelerator ";
        std::cout << std::endl;
        // device name
        status = clGetDeviceInfo(*i, CL_DEVICE_NAME,
                                 buf.size(), &buf[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_NAME)" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Name: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // device version
        status = clGetDeviceInfo(*i, CL_DEVICE_VERSION,
                                 buf.size(), &buf[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_VERSION)"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Version: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // device vendor
        status = clGetDeviceInfo(*i, CL_DEVICE_VENDOR,
                                 buf.size(), &buf[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_VENDOR)"
                      << std::endl; 
            exit(EXIT_FAILURE);
        }
        std::cout << "  Vendor: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // device profile
        status = clGetDeviceInfo(*i, CL_DEVICE_PROFILE,
                                 buf.size(), &buf[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_PROFILE)" 
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Profile: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // # compute units
        status = clGetDeviceInfo(*i, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(cl_uint), &d, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Compute units: " << d << std::endl;
        // # work item dimensions
        cl_uint maxWIDim = 0;
        status = clGetDeviceInfo(*i, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                 sizeof(cl_uint), &maxWIDim, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - "
                      << "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Max work item dim: " << maxWIDim << std::endl;
        // # work item sizes
        std::vector< size_t > wiSizes(maxWIDim, size_t(0));
        status = clGetDeviceInfo(*i, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                 sizeof(size_t)*wiSizes.size(),
                                 &wiSizes[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - "
                      << "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES)"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Work item sizes:";
        for(std::vector< size_t >::const_iterator s = wiSizes.begin();
            s != wiSizes.end(); ++s) {
            std::cout << ' ' << *s;
        }
        std::cout << std::endl;
        // max clock frequency
        status = clGetDeviceInfo(*i, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                 sizeof(cl_uint), &d, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - "
                      << "clGetDeviceInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY)"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Max clock freq: " << d << " MHz" << std::endl;
        // global memory
        cl_ulong m = 0;
        status = clGetDeviceInfo(*i, CL_DEVICE_GLOBAL_MEM_SIZE,
                                 sizeof(cl_ulong), &m, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Global memory: " << m << " bytes" << std::endl;
        // local memory
        m = 0;
        status = clGetDeviceInfo(*i, CL_DEVICE_LOCAL_MEM_SIZE,
                                 sizeof(cl_ulong), &m, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_SIZE)"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "  Local memory: " << m << " bytes" << std::endl;
    }
}


//------------------------------------------------------------------------------
void print_platforms() {
    cl_uint numPlatforms = 0;
    cl_platform_id platform = 0;
    cl_int status = clGetPlatformIDs(0, 0, &numPlatforms);
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR - clGetPlatformIDs()" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(numPlatforms < 1) {
        std::cout << "No OpenCL platform detected" << std::endl;
        exit(EXIT_SUCCESS);
    }
    typedef std::vector< cl_platform_id > PlatformIds;
    PlatformIds platforms(numPlatforms);
    status = clGetPlatformIDs(platforms.size(), &platforms[0], 0);
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR - clGetPlatformIDs()" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector< char > buf(0x10000, char(0));
    int p = 0;
    std::cout << "\n***************************************************\n";  
    std::cout << "Number of platforms: " << platforms.size() << std::endl;
    for(PlatformIds::const_iterator i = platforms.begin();
        i != platforms.end(); ++i, ++p) {
        
        std::cout << "\n-----------\n"; 
        std::cout << "Platform " << p << std::endl;
        std::cout << "-----------\n";  
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_VENDOR,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(EXIT_FAILURE);    
        }
        std::cout << "Vendor: " << &buf[ 0 ] << '\n'; 
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_PROFILE,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Profile: " << &buf[ 0 ] << '\n'; 
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_VERSION,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Version: " << &buf[ 0 ] << '\n';     
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_NAME,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Name: " << &buf[ 0 ] << '\n';  
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_EXTENSIONS,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Extensions: " << &buf[ 0 ] << '\n';
        print_devices(*i);
        std::cout << "\n===================================================\n"; 

    }
}