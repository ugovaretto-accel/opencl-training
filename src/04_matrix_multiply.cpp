//Matrix multiply example
//Author: Ugo Varetto
#include <iostream>
#include <cstdlib>
#include "clutil.h"

typedef float real_t;

//------------------------------------------------------------------------------
struct CLRunTimeEnv {
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue commandQueue;
};

//------------------------------------------------------------------------------
CLRunTimeResources creat_cl_rtenv(const std::string& platformName,
	                              const std::string& deviceType,
	                              int deviceNum,
	                              const char* clSourcePath,
	                              const char* kernelName, 
	                              const std::string& clSourcePrefix) {

    CLRunTimeEnv rt;

	//1)create context
    rt.context = create_cl_context(platformName, deviceType, deviceNum);
    
    //2)load kernel source
    const std::string programSource = clSourcePrefix 
                                      + "\n" 
                                      + load_text(clSourcePath);
    const char* src = programSource.c_str();
    const size_t sourceLength = programSource.length();

    //3)build program and create kernel
    cl_int status;
    rt.program = clCreateProgramWithSource(rt.context, //context
                                           1,   //number of strings
                                           &src, //lines
                                           &sourceLength, // size 
                                           &status);  // status 
    check_cl_error(status, "clCreateProgramWithSource");
    cl_device_id deviceID; //only a single device was selected
    // retrieve actual device id from context
    status = clGetContextInfo(rt.context,
                              CL_CONTEXT_DEVICES,
                              sizeof(cl_device_id),
                              &deviceID, 0);
    check_cl_error(status, "clGetContextInfo");
    cl_int buildStatus = clBuildProgram(rt.program, 1, &deviceID, 0, 0, 0);
    //log output if any
    char buffer[0x10000] = "";
    size_t len = 0;
    status = clGetProgramBuildInfo(rt.program,
                                   deviceID,
                                   CL_PROGRAM_BUILD_LOG,
                                   sizeof(buffer),
                                   buffer,
                                   &len);
    check_cl_error(status, "clBuildProgramInfo");
    if(len > 1) std::cout << "Build output: " << buffer << std::endl;
    check_cl_error(buildStatus, "clBuildProgram");

    const char* kernelName = argv[5];
    rt.kernel = clCreateKernel(rt.program, kernelName, &status);
    check_cl_error(status, "clCreateKernel"); 

    rt.commandQueue = clCreateCommandQueue(rt.context, deviceID, 0, &status);
    check_cl_error(status, "clCreateCommandQueue");

    return rt;
}

//------------------------------------------------------------------------------
void release_cl_rtenv(CLRunTimeEnv& rt) {
    check_cl_error(clReleaseCommandQueue(rt.commandQueue),
    	                                 "clReleaseCommandQueue");
    check_cl_error(clReleaseKernel(rt.kernel), "clReleaseKernel");
    check_cl_error(clReleaseProgram(rt.program), "clReleaseProgram");
    check_cl_error(clReleaseContext(rt.context), "clReleaseContext");
}

//------------------------------------------------------------------------------
std::vector< real_t > create_matrix(int cols, int rows) {
	std::vector< real_t > m(cols * rows);
	rand(time(0));
	for(std::vector<real_t>::iterator i = m.begin();
	    i != m.end(); ++i) *i = rand() % 10; 
	}
	return m;
}


//------------------------------------------------------------------------------
void host_matmul(const std::vector< real_t >& A,
	             const std::vector< real_t >& B,
	             std::vector< real_t >& C, 
	             int a_columns,
	             int b_columns) {
	const int rows = a_columns;
	const int columns = b_columns;
	for(int r = 0; r != rows; ++r) {
		for(int c = 0; c != columns; ++c) {
			C[r*b_columns + c] = 0;
			for(int ic = 0; ic != a_columns; ++ic) {
				C[r*b_columns + c] += A[r*a_columns + ic]
				                      * B[ic*b_colummns + c];
			}
		}
	}
}

//------------------------------------------------------------------------------
bool check_result(const std::vector< real_t >& v1,
	              const std::vector< real_t >& v2,
	              double eps) {
    for(int i = 0; i != v1.size(); ++i) {
    	if(double(std::abs(v1[i] - v2[i])) > eps) return false;
    }
    return true;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 6) {
        std::cout << "usage: " << argv[0]
                  << " <platform name> <device type = default | cpu | gpu "
                     "| acc | all>  <device num> <OpenCL source file path>"
                     " <kernel name>"
                  << std::endl;
        return 0; 
    }
    const double EPS = 0.00001;
    const int SIZE = 16; //16 x 16
    CLRunTimeEnv clenv = creat_cl_rtenv(argv[1], argv[2], atoi(argv[3]),
    	                                argv[4], argv[5],
    	                                "#define CACHE_ROWS 4\n"
    	                                "#define CACHE_COLUMNS 4\n");
   

    //create input and output matrices
    const std::vector<real_t> A = create_matrix(SIZE, SIZE);
    const std::vector<real_t> B = create_matrix(SIZE, SIZE);
    std::vector<real_t> C(SIZE * SIZE,real_t(0));
    std::vector<real_t> refC(SIZE * SIZE,real_t(0));        
    
    //allocate output buffer on OpenCL device
    const size_t BYTE_SIZE = SIZE * SIZE * sizeof(real_t);
    cl_mem devC = clCreateBuffer(clenv.context,
                                 CL_MEM_WRITE_ONLY,
                                 BYTE_SIZE,
                                 0,
                                 &status);
    check_cl_error(status, "clCreateBuffer");

    //allocate input buffers on OpenCL devices and copy data
    cl_mem devA = clCreateBuffer(clenv.context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 BYTE_SIZE,
                                 &A[0], //<-- copy data from A
                                 &status); 
    cl_mem devB = clCreateBuffer(clenv.context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 BYTE_SIZE,
                                 &B[0], //<-- copy data from A
                                 &status); 

    //set kernel parameters
    status = clSetKernelArg(clenv.kernel, //kernel
                            0,      //parameter id
                            sizeof(cl_mem), //size of parameter
                            &devA); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(A)");
    status = clSetKernelArg(clenv.kernel, //kernel
                            1,      //parameter id
                            sizeof(cl_mem), //size of parameter
                            &devB); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(B)");
    //set kernel parameters
    status = clSetKernelArg(clenv.kernel, //kernel
                            2,      //parameter id
                            sizeof(cl_mem), //size of parameter
                            &devC); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(C)");
    //set kernel parameters
    status = clSetKernelArg(clenv.kernel, //kernel
                            3,      //parameter id
                            sizeof(int), //size of parameter
                            &SIZE); //pointer to parameter
    check_cl_error(status, "clSetKernelArg(SIZE)");


    //7)setup kernel launch configuration
    //total number of threads == number of array elements
    const size_t globalWorkSize[2] = {SIZE, SIZE};
    //number of per-workgroup local threads
    const size_t localWorkSize[2] = {CACHE_SIZE, CACHE_SIZE}; 

    //8)launch kernel
    status = clEnqueueNDRangeKernel(clenv.commandQueue, //queue
                                    clenv.kernel, //kernel                                   
                                    2, //number of dimensions for work-items
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
    
    //9)read back and print results
    status = clEnqueueReadBuffer(clenv.commandQueue,
                                 devC,
                                 CL_TRUE, //blocking read
                                 0, //offset
                                 BYTE_SIZE, //byte size of data
                                 &C[0], //destination buffer in host memory
                                 0, //number of events that need to
                                    //complete before transfer executed
                                 0, //list of events that need to complete
                                    //before transfer executed
                                 0); //event identifying this specific operation
    check_cl_error(status, "clEnqueueReadBuffer");
    
    host_matmul(A, B, refC);

    if(check_result(refC, C, EPS) {
    	std::cout << "PASSED" << std::endl;
    } else {
    	std::cout << "FAILED" << std::endl;
    }	

    release_cl_rtenv(clenv);
   
    return 0;
}