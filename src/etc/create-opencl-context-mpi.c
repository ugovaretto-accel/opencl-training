#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <CL/cl.h>

int main( int argc, char** argv ) {
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint platforms, devices;
    int rank = -1;


    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  
    char* env = getenv( "CRAY_CUDA_PROXY" );
    if( env ) printf( "Process %d: CRAY_CUDA_PROXY = %s\n", rank, env); 
    else printf( "Process %d: CRAY_CUDA_PROXY not set\n", rank);
    
    error=clGetPlatformIDs(1, &platform, &platforms);
    if (error != CL_SUCCESS) {
        printf("Process %d: Error number %d\n", rank, error);
        MPI_Finalize();
        return -1;
    }
    error=clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &devices);
    if(error != CL_SUCCESS) {
        printf("Process %d: Error number %d\n", rank, error);
        MPI_Finalize();
        return -1;
    }
    cl_context_properties properties[]={
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0};
    
    cl_context context=clCreateContext(properties, 1, &device, NULL, NULL, &error);
    if(error != CL_SUCCESS) {
        printf("Process %d: Cannot create CL context - Error number %d\n", rank, error); 
        MPI_Finalize();
        return -1;
    }
    sleep(10);
    printf("Process %d: OK\n", rank);
    MPI_Finalize();
    return 0;   
}
