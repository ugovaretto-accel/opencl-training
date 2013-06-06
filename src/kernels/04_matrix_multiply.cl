#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#else
typedef float real_t;
#endif
//square matrices only
//TILE_SIZE and DOUBLE are defined from outside the kernel
//by prefixing this code with proper #define statements from within
//the driver program;
//launch with grid = [columns X rows]
//in order to take advantage of caching and coalescing the
//elements which are contiguous in memory must map to the
//fastest moving index i.e. for row major matrices the column
//index must map to the x coordinate of the launch grid

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
real_t get_matrix_element(global const real_t* m, //matrix
                          int tileSize,    //tile size
                          int blockCol,    //column index of output block 
                          int blockRow,    //row index of output row
                          int col,         //local column index of block element
                          int row,         //local row index of block element 
                          int num_columns  //number of columns of matrix 'm'
                          ) {                                           
  
    return m[( blockRow * tileSize + row ) * num_columns
           + blockCol * tileSize + col];

}

//------------------------------------------------------------------------------
//work item size must be exactly TILE_SIZE x TILE_SIZE
kernel void block_matmul(global const real_t* A,
                         global const real_t* B,
                         global real_t* C,
                         int columns) {

    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int blockRow = get_group_id(1);
    const int blockCol = get_group_id(0);
	__local real_t a[TILE_SIZE][TILE_SIZE];
	__local real_t b[TILE_SIZE][TILE_SIZE]; 
    real_t out = 0;
    //iterate over rows of block in a and columns of blocks in b
    for( int blockId = 0; blockId != columns / TILE_SIZE; ++blockId ) {
        //copy data into shared memory
        a[row ][col] = 
        	get_matrix_element(A, TILE_SIZE,
        	                   blockId, // <-- column id of block
        	                   blockRow,// <-- row id of block
        	                   col, row, columns);
        b[row ][col] = 
        	get_matrix_element(B, TILE_SIZE,
        	                   blockCol, // <-- column id of block
        	                   blockId,  // <-- row id of block
        	                   col, row, columns );
        // barrier required to guarantee that data are computed before next step
        // where a thread accesses data computed by other threads
        barrier(CLK_LOCAL_MEM_FENCE) 
        for( int k = 0; k != TILE_SIZE; ++k ) {
            out += a[ row ][ k ] * b[ k ][ col ];           
        }
        barrier(CLK_LOCAL_MEM_FENCE)  
    }
    C[(blockRow * TILE_SIZE + row) * columns
       + blockCol * TILE_SIZE + col ] = out;     
}

