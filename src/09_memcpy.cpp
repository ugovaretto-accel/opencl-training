//Memcopy example w/ bandwidth tests.
//Author: Ugo Varetto
//Note: page-locked memory transfers might not work properly on systems
//sharing the same memory for both host and device (e.g. CPU)
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <stdexcept>

//standard OpenCL C++ wrapper include:
//http://www.khronos.org/registry/cl/api/1.1/cl.hpp
#include "cl.hpp"

typedef float real_t;

typedef std::vector< char > ByteArray;

//------------------------------------------------------------------------------
double copy_host_to_device(const ByteArray& data,
                           const cl::Context& context, 
                           cl::CommandQueue& queue) {
   cl::Buffer buffer(context, 
                     CL_MEM_READ_ONLY,
                     data.size(),
                     0);
   cl::Event profileEvent;
   queue.finish();
   queue.enqueueWriteBuffer(buffer,
                            CL_TRUE,
                            0,
                            data.size(),
                            &data[0],
                            0,
                            &profileEvent);

   // Configure event processing
   const cl_ulong start = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_START>();
   const cl_ulong end = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_END>();
   return double(end - start) / 1E6;                        
}

//------------------------------------------------------------------------------
double copy_host_to_device_page_locked(const ByteArray& data,
                                       const cl::Context& context, 
                                       cl::CommandQueue& queue) {

   //have OpenCL allocate a page-locked buffer with CL_MEM_ALLOC_HOST_PTR  
   cl::Buffer buffer(context, 
                     CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     data.size(), 0);
   cl::Event profileEvent;
   queue.finish();
   //map buffer to host memory: this might trigger a device-host transfer;
   //in this case since no host data was transfered to the CL buffer, no
   //device --> host transfer should occur 
   void* hostPtr = queue.enqueueMapBuffer(buffer,
                            CL_TRUE,
                            CL_MAP_WRITE,
                            0,
                            data.size(),
                            0,
                            0);
   if(hostPtr == 0) throw std::runtime_error("ERROR - NULL host pointer");
   //the unmap function is what triggers the actual host --> device memory
   //transfer to keep host and device memory in sync; note that if both
   //host and device do use the same memory space the timings might be
   //meaningless and represent only the latency of the C function call itself
   queue.enqueueUnmapMemObject(buffer, hostPtr, 0, &profileEvent);

   // Configure event processing
   const cl_ulong start = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_START>();
   const cl_ulong end = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_END>();
   return double(end - start) / 1E6;                        
}


//------------------------------------------------------------------------------
double copy_device_to_host(ByteArray& data,
                           const cl::Context& context, 
                           cl::CommandQueue& queue) {
   cl::Buffer buffer(context, 
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     data.size(),
                     &data[0]);
   cl::Event profileEvent;
   queue.finish();
   queue.enqueueReadBuffer(buffer,
                           CL_TRUE,
                           0,
                           data.size(),
                           &data[0],
                           0,
                           &profileEvent);

   // Configure event processing
   const cl_ulong start = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_START>();
   const cl_ulong end = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_END>();
   return double(end - start) / 1E6;                        
}

//------------------------------------------------------------------------------
double copy_device_to_host_page_locked(const ByteArray& data,
                                       const cl::Context& context, 
                                       cl::CommandQueue& queue) {

  
   cl::Buffer buffer(context, 
                     CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     data.size(), 0);
   cl::Event profileEvent;
   queue.finish();

   //actual data transfer operations do happen at map/unmap time

   //in order to force an actual device --> host transfer we need to somehow
   //"touch" the device memory by first enqueueing a host --> device 
   //transfer through unmap; not doing this might not trigger a 
   //device --> host transfer when mapping memory


   //dummy host --> device transfer - not timed
   void* hostPtr = queue.enqueueMapBuffer(buffer,
                            CL_TRUE,
                            CL_MAP_WRITE,
                            0,
                            data.size(),
                            0,
                            0);
   queue.enqueueUnmapMemObject(buffer, hostPtr, 0, 0);
   //device to host transfer
   hostPtr = queue.enqueueMapBuffer(buffer,
                            CL_TRUE,
                            CL_MAP_WRITE,
                            0,
                            data.size(),
                            0,
                            &profileEvent);
   queue.enqueueUnmapMemObject(buffer, hostPtr, 0, 0);
   if(hostPtr == 0) throw std::runtime_error("ERROR - NULL host pointer");

   // Configure event processing
   const cl_ulong start = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_START>();
   const cl_ulong end = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_END>();
   return double(end - start) / 1E6;                        
}

//------------------------------------------------------------------------------
double copy_device_to_device(const ByteArray& data,
                             const cl::Context& context, 
                             cl::CommandQueue& queue) {
   cl::Buffer src(context, 
                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                  data.size(),
                  const_cast< char* >(&data[0]));
   cl::Buffer dest(context,
                   CL_MEM_WRITE_ONLY,
                   data.size(),
                   0);

   cl::Event profileEvent;
   queue.finish();
   queue.enqueueCopyBuffer(src,
                           dest,
                           0,
                           0,
                           data.size(),
                           0,
                           &profileEvent);
   queue.finish();
   // Configure event processing
   const cl_ulong start = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_START>();
   const cl_ulong end = profileEvent
                           .getProfilingInfo<CL_PROFILING_COMMAND_END>();
   return double(end - start) / 1E6;                        
}



//------------------------------------------------------------------------------
double GBs(size_t sizeInBytes, double timeInSeconds) {
   return (double(sizeInBytes) / 0x40000000) / timeInSeconds; 
}


//------------------------------------------------------------------------------
int main(int argc, char** argv) {
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   if(argc < 5) {
      std::cout << "usage: " << argv[0]
                << " <platform id(0, 1...)>"
                   " <device type: default | cpu | gpu | acc>"
                   " <device id(0, 1...)"
                   " <size>"
                   " [page-locked]"

                << std::endl; 
      exit(EXIT_FAILURE);          
   }
   const bool pageLocked = argc > 5 ? true : false;
   const int platformID = atoi(argv[1]);
   cl_device_type deviceType;
   const std::string dt(argv[2]);
   if(dt == "default") deviceType = CL_DEVICE_TYPE_DEFAULT;
   else if(dt == "cpu") deviceType = CL_DEVICE_TYPE_CPU;
   else if(dt == "gpu") deviceType = CL_DEVICE_TYPE_GPU;
   else if(dt == "acc") deviceType = CL_DEVICE_TYPE_ACCELERATOR;
   else {
      std::cerr << "ERROR - unrecognized device type " << dt << std::endl;
      exit(EXIT_FAILURE);
   } 
   const int deviceID = atoi(argv[3]);
   const size_t SIZE = atoll(argv[4]);
   try {
      cl::Platform::get(&platforms);
      if(platforms.size() <= platformID) {
         std::cerr << "Platform id " << platformID << " is not available\n";
         exit(EXIT_FAILURE);
      }
      platforms[platformID].getDevices(deviceType, &devices);
      cl::Context context(devices);
      
      //create command queue to use for copy operations
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      ByteArray data(SIZE);
      const double h2d = copy_host_to_device(data, context, queue);
      const double d2h = copy_device_to_host(data, context, queue);
      const double d2d = copy_device_to_device(data, context, queue);                                          
      const double h2dpl = !pageLocked ? 0 :
                        copy_host_to_device_page_locked(data, context, queue);
      const double d2hpl = !pageLocked ? 0 :
                        copy_device_to_host_page_locked(data, context, queue);

      std::cout << "Elapsed time (ms):" << std::endl
                << "  host to device:               " << h2d   << std::endl
                << "  device to host:               " << d2h   << std::endl
                << "  device to device:             " << d2d   << std::endl;
      if(pageLocked) {          
         std::cout
                << "  host to device - page locked: " << h2dpl << std::endl
                << "  device to host - page locked: " << d2hpl << std::endl;
      }          
      std::cout << "Bandwidth(GB/s):" << std::endl
                << "  host to device:               " 
                << GBs(SIZE, h2d / 1E3) << std::endl
                << "  device to host:               " 
                << GBs(SIZE, d2h / 1E3) << std::endl
                << "  device to device:             " 
                << 2 * GBs(SIZE, d2d / 1E3)
                << std::endl;
      if(pageLocked) {
         std::cout
                << "  host to device - page locked: " << GBs(SIZE, h2dpl / 1E3)
                << std::endl
                << "  device to host - page locked: " << GBs(SIZE, d2hpl / 1E3)
                << std::endl;
      }          

   } catch(cl::Error e) {
      std::cerr << e.what() << ": Error code " << e.err() << std::endl;   
      exit(EXIT_FAILURE);
   } catch(const std::exception& e) {
      std::cerr << e.what() << std::endl;
      exit(EXIT_FAILURE);
   }
   return 0;
}