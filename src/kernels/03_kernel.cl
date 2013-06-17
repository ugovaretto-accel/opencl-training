//set elements of array to value
//Author: Ugo Varetto
#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#else
typedef float real_t;
#endif
__kernel void arrayset(__global real_t* outputArray,
	                   real_t value) {
	//get global thread id for dimension 0
	const int id = get_global_id(0);
	outputArray[id] = value; 
}
