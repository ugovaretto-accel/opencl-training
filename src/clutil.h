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
//the following function only fills the requested CLEnv fields:
//context and command queue are always reaturned; program and
//kernel are returned only if the source path and kernel name are
//not NULL
CLEnv create_clenv(const std::string& platformName,
                   const std::string& deviceType,
                   int deviceNum,
                   bool enableProfiling = false,
                   const char* clSourcePath = 0,
                   const char* kernelName = 0, 
                   const std::string& clSourcePrefix = std::string(),
                   const std::string& buildOptions = std::string());
void release_clenv(CLEnv& e);
//executes kernel synchronously and returns elapsed time in milliseconds
double timeEnqueueNDRangeKernel(cl_command_queue command_queue,
                                cl_kernel kernel,
                                cl_uint work_dim,
                                const size_t *global_work_offset,
                                const size_t *global_work_size,
                                const size_t *local_work_size,
                                cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list);
