//C++ example: shows how little code is required (in fact less than in CUDA C),
//when using OpenCL through the C++ api
//Author: Ugo Varetto
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <cstdlib>

//standard cl C++ wrapper include:
//http://www.khronos.org/registry/cl/api/1.1/cl.hpp
#include "cl.hpp"

typedef float real_t;

int main(int argc, char** argv) {
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   cl::Event profileEvent;
   cl_ulong start, end;
   const size_t SIZE = 10;
   if(argc < 5) {
      std::cout << "usage: " << argv[0]
                << " <platform id(0, 1...)>"
                   " <device type: default | cpu | gpu | acc>"
                   " <OpenCL source>"
                   " <kernel function name>\n"
                   "Must use kernel that sets all the element of an array to"
                   " a value and accepts an output buffer and an int"
                << std::endl; 
      exit(EXIT_FAILURE);          
   }
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
   

   try {
      // Place the GPU devices of the first platform into a context
      cl::Platform::get(&platforms);
      if(platforms.size() <= platformID) {
         std::cerr << "Platform id " << platformID << " is not available\n";
         exit(EXIT_FAILURE);
      }
      platforms[platformID].getDevices(deviceType, &devices);
      cl::Context context(devices);
      
      // Create kernel
      std::ifstream programFile(argv[3]);
      std::string programSrc(std::istreambuf_iterator<char>(programFile),
            (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(programSrc.c_str(),
                                  0)); // 0 means that the source strings
                                       // are NULL terminated
      cl::Program program(context, source);
      program.build(devices);
      cl::Kernel kernel(program, kernelName.c_str());

      // Create buffer and make it a kernel argument
      std::vector< real_t > data(SIZE, 0);
      cl::Buffer buffer(context, 
                        CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(real_t) * SIZE,
                        &data[0]);
      kernel.setArg(0, buffer);
      kernel.setArg(1, 3.14f); //set all elements of array to 3.14

      // Enqueue kernel-execution command with profiling event
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
      queue.enqueueNDRangeKernel(kernel,
                                 cl::NDRange(0),
                                 cl::NDRange(SIZE),
                                 cl::NDRange(1),
                                 0, // wait events *
                                 &profileEvent);
      queue.finish();

      // Configure event processing
      start = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      end = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      std::cout << "Elapsed time: " << (end - start) << " ns." << std::endl;

      //read and print data
      queue.enqueueReadBuffer(buffer,
                              CL_TRUE, //blocking read
                              0, //offset
                              sizeof(real_t) * SIZE, //byte size of data
                              &data[0], //destination buffer in host memory
                              0, //events that need to
                                 //complete before transfer executed
                              0); //event identifying this specific operation
      std::ostream_iterator< real_t > oit(std::cout, " ");
      std::copy(data.begin(), data.end(), oit);
      std::cout << std::endl;

   }
   catch(cl::Error e) {
      std::cout << e.what() << ": Error code " << e.err() << std::endl;   
   }
   return 0;
}