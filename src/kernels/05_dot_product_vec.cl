//Partial parallel dot product with vector types.
//Author: Ugo Varetto

#define VEC_TYPE_DEF(t,n) typedef t##n vec_real_t 

//Subdivides input array into sub-arrays and
//performs dot product on each sub-array; the dot product of each sub-array
//is copied into the output buffer; a final reduction step has to be performed
//on the host.

//declare data types

#ifdef DOUBLE

#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#if VEC_WIDTH == 1
typedef double vec_real_t;
#elif VEC_WIDTH == 4
VEC_TYPE_DEF(double, 4);
#elif VEC_WIDTH == 8
VEC_TYPE_DEF(double, 8);
#else
#error VEC_WIDTH 
#endif 

#else

typedef float real_t;
#if VEC_WIDTH == 1
typedef float vec_real_t;
#elif VEC_WIDTH == 4
VEC_TYPE_DEF(float, 4);
#elif VEC_WIDTH == 8
VEC_TYPE_DEF(float, 8);
#else
#error VEC_WIDTH 
#endif 

#endif

// define SUM function for results of * operation on vector data types

#if VEC_WIDTH == 1
#define VEC_SUM(r) r
#elif VEC_WIDTH == 4
#define VEC_SUM(r) r[0] + r[1] + r[2] + r[3]
#elif VEC_WIDTH == 8
#define VEC_SUM(r) r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]
#endif


//BLOCK_SIZE and DOUBLE are defined from outside the kernel by prefixing
//the source code witha a "#define BLOCK_SIZE" and "#define DOUBLE"
//statement from within the driver program
__kernel void dotprod(__global const vec_real_t* v1,
                      __global const vec_real_t* v2,
                      __global real_t* reduced) {

    __local real_t cache[BLOCK_SIZE];

    const int cache_idx = get_local_id(0);
    const int id = get_global_id(0);
    //copy data into buffer shared by all work items(threads)
    //in a workgroup
    const vec_real_t r = v1[id] * v2[id];
    cache[cache_idx] = VEC_SUM(r);
    //barrier to guarantee that all elements are copied
    //before performing actual reduction
    barrier(CLK_LOCAL_MEM_FENCE); 
    //iterate over buffer: at each step do sum
    //element mapped to current work item with
    //element at position current + step;
    //at each step only the work items with an id < step
    //are active 
    int step = BLOCK_SIZE / 2;
    while(step > 0) {
    	if(cache_idx < step) {
    	    cache[cache_idx] += cache[cache_idx + step];
        }
        //need to synchronize execution to make sure that
        //element at position cache[cache_idx + step]
        //is up to date
        barrier(CLK_LOCAL_MEM_FENCE); 
    	step /= 2;
    }
    //local work item 0 takes care of copying the data into
    //the output buffer at position equal to this workgroup id
    if(cache_idx == 0) reduced[get_group_id(0)] = cache[0];
}