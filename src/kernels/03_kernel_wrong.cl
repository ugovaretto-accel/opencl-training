//set elements of array to value;
//error introduced on purpose to show output
//of OpenCL compiler
#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#else
typedef float real_t;
#endif
__kernel void arrayset(__global real_t* outputArray,
	                   real_t value) {
	//get global thread id for dimension 0
	const int i = get_global_id(0);
	//WARNING: error introduced to show output from OpenCL compiler
	outputArray[id] = value; 
}