#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#else
typedef float real_t;
#endif
//CACHE_ROWS, CACHE_ROWS and DOUBLE are defined from outside the kernel
//by prefixing this code with proper #define statements from within
//the driver program;
//launch with grid = [columns X rows]
//in order to take advantage of caching and coalescing the
//elements which are contiguous in memory must map to the
//fastest moving index i.e. for row major matrices the column
//index must map to the x coordinate of the launch grid  
kernel void matmul(global const real_t* A,
                   global const real_t* B,
                   global real_t* C,
                   int a_columns,
                   int b_columns) {
    const int column = get_global_id(0);
    const int row    = get_global_id(1);
    real_t e = 0;
    for(int c = 0; c != a_columns; ++c) {
         e +=  A[row * a_columns + c] * B[c * b_columns + column ]; 
    }
    C[row * b_columns + column] = e;
}

//square matrices only
//work item size must be exactly CACHE_COLS X CACHE_ROWS
kernel void block_matmul(global const real_t* A,
                         global const real_t* B,
                         global real_t* C,
                         int columns,
                         int b_columns) {
    __local real_t a[CACHE_ROWS][CACHE_COLUMNS];
    __local real_t b[CACHE_ROWS][CACHE_COLUMNS];
    const int lcol = get_local_id(0);
    const int lrow = get_local_id(1);
    const int globalIdx = get_global_id(1) * columns + get_global_id(0);
    a[lrow][lcol] = A[globalIdx];
    b[lrow][lcol] = B[globalIdx];
    barrier(CLK_LOCAL_MEM_FENCE); 
    real_t e = (real_t) 0;
    for(int c = 0; c != CACHE_COLUMNS; ++c) {
        e += a[lrow][c] * b[c][lcol]; 
    }
    C[globalIdx] = e;
}

//generic implementation of block mat-mat multiply where
//launch grid size id smaller than the output matrix size:
//have each thread iterate over a number of elements
//work item size must be exactly CACHE_COLS X CACHE_ROWS
kernel void block_matmul_generic(global const real_t* A,
                                 global const real_t* B,
                                 global real_t* C,
                                 int a_columns,
                                 int b_columns) {
    __local real_t a[CACHE_ROWS][CACHE_COLUMNS];
    __local real_t b[CACHE_ROWS][CACHE_COLUMNS];
    const int rows = a_columns;
    const int columns = b_columns;
    const int lcol = get_local_id(0);
    const int lrow = get_local_id(1);
    //in case the launch grid is smaller than the actual data size,
    //as it happens in many cases, the kernel is required to iterate
    //over the entire data set
    const int gCols = get_global_size(0);
    const int gRows = get_global_size(1);
    for(int gc = 0; gc < columns; gc += gCols) {
    	const int col = (get_global_id(0) + gc);
    	for(int gr = 0; gr < rows; gr += gRows) {
    		const int row = (get_global_id(1) + gr);
    		const int globalIdx = row * columns + col;
    		if(row >= rows || col >= columns ) {
    			a[lrow][lcol] = (real_t) 0;
        	    b[lrow][lcol] = (real_t) 0;
        	    return;
    		} else {
        	    a[lrow][lcol] = A[row * a_columns + col];
        	    b[lrow][lcol] = B[row * b_columns + col];
    		}
    		barrier(CLK_LOCAL_MEM_FENCE);
    		real_t e = (real_t) 0;
    		for(int c = 0; c != CACHE_COLUMNS; ++c) {
        		e += a[lrow][c] * b[c][lcol]; 
    		}
    		if( gc == 0 && gr == 0) C[globalIdx] = e;
    		else C[globalIdx] += e;
    	}
    }
}