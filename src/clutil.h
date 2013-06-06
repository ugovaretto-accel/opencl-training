#pragma once
//Utility functions
//Author: Ugo Varetto
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif




struct CLEnv {
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue commandQueue;
};

void check_cl_error(cl_int status, const char* msg);
cl_context create_cl_context(const std::string& platformName,
                             const std::string& deviceTypeName,
                             int deviceNum);
std::string load_text(const char* filepath);
cl_device_id get_device_id(cl_context ctx);
void print_platforms();
CLEnv create_clenv(const std::string& platformName,
                   const std::string& deviceType,
                   int deviceNum,
                   bool enableProfiling,
                   const char* clSourcePath,
                   const char* kernelName, 
                   const std::string& clSourcePrefix);
void release_clenv(CLEnv& e);