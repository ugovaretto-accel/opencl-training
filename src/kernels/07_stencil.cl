//Convolution with and without image objects
//Author: Ugo Varetto

#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#else
typedef float real_t;
#endif

//------------------------------------------------------------------------------
__kernel void filter(const __global real_t* src,
                     int size,
                     const __global real_t* filter,
                     int filterSize,
                     __global real_t* out ) {
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    if(coord.x < filterSize / 2
       || coord.x >= (size - filterSize / 2)
       || coord.y < filterSize / 2
       || coord.y >= (size - filterSize / 2)) return; 
    real_t e = (real_t) 0;
    for(int i = 0; i < filterSize / 2; ++i) {
        for(int j = 0; j < filterSize / 2; ++j) {
            e += src[(coord.y + i) * size + coord.x + j]
                 * filter[(i + filterSize / 2) * filterSize + j
                          + filterSize / 2]; 
        }
    }
    out[coord.y * size + coord.x] = e;
}



//------------------------------------------------------------------------------
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
                               | CLK_ADDRESS_CLAMP
                               | CLK_FILTER_NEAREST;

__kernel void filter_image(read_only image2d_t src,
                           read_only image2d_t filter,
                           __global real_t* out) {
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const int width = get_image_width(src);
    const int height = get_image_height(filter);
    const int fwidth = get_image_width(filter);
    const int fheight = get_image_height(filter);
    if(coord.x < fwidth / 2
       || coord.x >= (width - fwidth / 2)
       || coord.y < fheight / 2
       || coord.y >= (height - fheight / 2))  return; 
    const float4 i = read_imagef(src, sampler, coord);
    float e = 0.0f;
    for(int i = 0; i < fwidth / 2; ++i) {
        for(int j = 0; j < fheight / 2; ++j) {
            const float4 weight = read_imagef(filter, sampler,
                                    (int2)(i + fheight / 2, j + fwidth / 2));
            const float4 iv = read_imagef(filter, sampler, 
                                          coord + (int2)(i, j));
           e += iv.x * weight.x; 
        }
    }
    out[coord.y * width + coord.x] = e;
}