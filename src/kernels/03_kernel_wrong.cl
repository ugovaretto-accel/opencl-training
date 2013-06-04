typedef float real_t;
kernel void arrayset(global real_t* outputArray,
	                 real_t value) {
	//get global thread id for dimension 0
	const int i = get_global_id(0);
	//WARNING: error introduced to show output from OpenCL compiler
	outputArray[id] = value; 
}