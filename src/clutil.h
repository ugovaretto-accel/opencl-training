#pragma once
//Utility functions
//Author: Ugo Varetto
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void check_cl_error(cl_int status, const char* msg);
cl_context create_cl_context(const std::string& platformName,
                             const std::string& deviceTypeName,
                             int deviceNum);
std::string load_text(const char* filepath);
cl_device_id get_device_id(cl_context ctx);
void print_platforms();