// Print OpenCL platform and device information.
// Author: Ugo Varetto

#include <iostream>
#include <vector>
#include <cstdlib>
#include <CL/cl.h>


//------------------------------------------------------------------------------
void print_devices(cl_platform_id pid) {
    cl_uint numDevices = 0;
    cl_int status = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 
                                   0, 0, &numDevices );
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR - clGetDeviceIDs" << std::endl;
        exit(-1);
    }
    if(numDevices < 1) return;
    
    typedef std::vector< cl_device_id > DeviceIds;
    DeviceIds devices(numDevices);
    status = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL,
                            devices.size(), &devices[ 0 ], 0 );
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR - clGetDeviceIDs" << std::endl;
        exit(-1);
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
            exit(-1);
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
            exit(-1);
        }
        std::cout << "  Name: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // device version
        status = clGetDeviceInfo(*i, CL_DEVICE_VERSION,
                                 buf.size(), &buf[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_VERSION)" << std::endl;
            exit(-1);
        }
        std::cout << "  Version: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // device vendor
        status = clGetDeviceInfo(*i, CL_DEVICE_VENDOR,
                                 buf.size(), &buf[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_VENDOR)" << std::endl;
            exit(-1);
        }
        std::cout << "  Vendor: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // device profile
        status = clGetDeviceInfo(*i, CL_DEVICE_PROFILE,
                                 buf.size(), &buf[0], 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_PROFILE)" << std::endl;
            exit(-1);
        }
        std::cout << "  Profile: " << &buf[0] << std::endl; 
        std::fill(buf.begin(), buf.end(), char(0));
        // # compute units
        status = clGetDeviceInfo(*i, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(cl_uint), &d, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)"
                      << std::endl;
            exit(-1);
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
            exit(-1);
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
            exit(-1);
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
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY)"
                      << std::endl;
            exit(-1);
        }
        std::cout << "  Max clock freq: " << d << " MHz" << std::endl;
        // global memory
        cl_ulong m = 0;
        status = clGetDeviceInfo(*i, CL_DEVICE_GLOBAL_MEM_SIZE,
                                 sizeof(cl_ulong), &m, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)"
                      << std::endl;
            exit(-1);
        }
        std::cout << "  Global memory: " << m << " bytes" << std::endl;
        // local memory
        m = 0;
        status = clGetDeviceInfo(*i, CL_DEVICE_LOCAL_MEM_SIZE,
                                 sizeof(cl_ulong), &m, 0);
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_SIZE)"
                      << std::endl;
            exit(-1);
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
        exit(-1);
    }
    if(numPlatforms < 1) {
        std::cout << "No OpenCL platform detected" << std::endl;
        exit(0);
    }
    typedef std::vector< cl_platform_id > PlatformIds;
    PlatformIds platforms(numPlatforms);
    status = clGetPlatformIDs(platforms.size(), &platforms[0], 0);
    if(status != CL_SUCCESS) {
        std::cerr << "ERROR - clGetPlatformIDs()" << std::endl;
        exit(-1);
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
            exit(-1);    
        }
        std::cout << "Vendor: " << &buf[ 0 ] << '\n'; 
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_PROFILE,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(-1);
        }
        std::cout << "Profile: " << &buf[ 0 ] << '\n'; 
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_VERSION,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(-1);
        }
        std::cout << "Version: " << &buf[ 0 ] << '\n';     
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_NAME,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(-1);
        }
        std::cout << "Name: " << &buf[ 0 ] << '\n';  
        status = ::clGetPlatformInfo(*i, CL_PLATFORM_EXTENSIONS,
                                     buf.size(), &buf[ 0 ], 0 );
        if(status != CL_SUCCESS) {
            std::cerr << "ERROR - clGetPlatformInfo(): " << std::endl;
            exit(-1);
        }
        std::cout << "Extensions: " << &buf[ 0 ] << '\n';
        print_devices(*i);
        std::cout << "\n===================================================\n"; 

    }
}

//------------------------------------------------------------------------------
int main(int, char**) {
    print_platforms();
    return 0;
}