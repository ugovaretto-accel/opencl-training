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
                   int columns) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    real_t e = 0;
    for(int c = 0; c != columns; ++c) {
         e +=  A[row * columns + c] * B[c * columns + col ]; 
    }
    C[row * columns + col] = e;
}

real_t get_matrix_element(global const real_t* m, //matrix
                          int blockDim,    //tile size
                          int blockCol,    //column index of output block 
                          int blockRow,    //row index of output row
                          int col,         //local column index of block element
                          int row,         //local row index of block element 
                          int num_columns  //number of columns of matrix 'm'
                          ) {                                           
  
    return m[( blockRow * blockDim + row ) * num_columns
           + blockCol * blockDim + col];

}

//square matrices only
//work item size must be exactly CACHE_COLS X CACHE_ROWS
kernel void block_matmul(global const real_t* A,
                         global const real_t* B,
                         global real_t* C,
                         int columns) {
    real_t out = 0.f;
    for( int b = 0; b != columns / TILE_SIZE; ++b ) {
        //copy data into shared memory
        M1[row ][col] = 
        	get_matrix_element(A, TILE_SIZE, b, blockRow, 
        	                   col, row, m1_columns);
        M2[row ][col] = 
        	get_matrix_element(B, TILE_SIZE, blockCol, b,
        	                   col, row, m2_columns );
        // barrier required to guarantee that data are computed before next step
        // where a thread accesses data computed by other threads
        barrier(CLK_LOCAL_MEM_FENCE) 
        for( int k = 0; k != TILE_SIZE; ++k ) {
            out += M1[ row ][ k ] * M2[ k ][ col ];           
        }
        barrier(CLK_LOCAL_MEM_FENCE)  
    }
    C[(blockRow * blockDim + row) * columns + blockCol * blockDim + col ] = out;     
}

