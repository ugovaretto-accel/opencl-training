//Load and execute a simple kernel which sets each array element to a specific
//value
//Author: Ugo Varetto
//Launch as e.g.:
// ./03 "AMD Accelerated Parallel Processing" default 0 \
//       src/kernels/03_kernel.cl arrayset
//Also try with 03_kernel_wrong.cl to see error output from OpenCL compiler 


#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "clutil.h"

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 4) {
        std::cout << "usage: " << argv[0]
                  << " <platform name>"
                     " <device type = default | cpu | gpu | acc | all>"
                     " <OpenCL source file path>"            
                  << std::endl;
        return 0; 
    }

    std::string platformName = argv[ 1 ];
    std::string deviceType = argv[2]; 
    std::string options;
    for(int a = 4; a != argc; ++a) {
        options += argv[a];
    }

    std::cout << options << std::endl;
    //create context
    cl_context ctx = create_cl_context(platformName, deviceType, 0);
     
    //load kernel source
    const std::string programSource = load_text(argv[3]);
    const char* src = programSource.c_str();
    const size_t sourceLength = programSource.length();

    //build program and create kernel
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
    cl_int buildStatus = clBuildProgram(program, 1, &deviceID, options.c_str(),
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
    if(len > 2) std::cout << "Build output: " << buffer << std::endl;
    check_cl_error(buildStatus, "clBuildProgram");
    
    size_t programSize = 0; 
    status = clGetProgramInfo(program,
                              CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t),
                              &programSize, //only a single device supported
                              0);
    std::cout << programSize << std::endl;
    check_cl_error(status, "clGetProgramInfo");
    char** binaries = 0;
    binaries = new char*[1];
    binaries[0] = new char[programSize]; 
    size_t returnedSize = 0; 
    status = clGetProgramInfo(program,
                              CL_PROGRAM_BINARIES,
                              sizeof(size_t),
                              binaries,
                              0);
                
    check_cl_error(status, "clGetProgramInfo");
    const std::string outputFileName(std::string(argv[3]) + ".clbin");
    std::ofstream out(outputFileName.c_str(), std::ios::out | std::ios::binary);
    if(!out) {
        std::cerr << "Error opening output file "
                  << outputFileName << std::endl;
        exit(EXIT_FAILURE);          
    }
    out.write(binaries[0], programSize);
    delete [] binaries[0];
    delete [] binaries;
    //release resources
    check_cl_error(clReleaseProgram(program), "clReleaseProgram");
    check_cl_error(clReleaseContext(ctx), "clReleaseContext");
    return 0;
}


