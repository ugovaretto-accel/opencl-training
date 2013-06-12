//Convolution with and without image objects; square grids and square filters
//Author: Ugo Varetto

//IMPORTANT: the core space size(total size - filter size) *must* be evenly
//divisible by the workgroup size in each dimension

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
    const int2 coord = (int2)(get_global_id(0) + filterSize / 2,
                              get_global_id(1) + filterSize / 2);
    real_t e = (real_t) 0;
    for(int i = -filterSize / 2; i <= filterSize / 2; ++i) {
        for(int j = -filterSize / 2; j <= filterSize / 2; ++j) {
            e += src[(coord.y + i) * size + coord.x + j]
                 * filter[(i + filterSize / 2) * filterSize + j
                          + filterSize / 2]; 
        }
    }
    out[coord.y * size + coord.x] = e / (filterSize * filterSize);
}

//------------------------------------------------------------------------------
//the following configuration is *required* when working with single element
//floating point values
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_FILTER_NEAREST |
                               CLK_ADDRESS_NONE;

#ifdef WRITE_TO_IMAGE
__kernel void filter_image(read_only image2d_t src,
                           read_only image2d_t filter,
                           write_only image2d_t out) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const int width = get_image_width(src);
    const int fwidth = get_image_width(filter);
    const int fheight = get_image_height(filter);
    coord += (int2)(fwidth / 2, fheight / 2);
    float e = 0.0f;
    for(int i = -fheight / 2; i <= fheight / 2; ++i) {
        for(int j = -fwidth / 2; j <= fwidth / 2; ++j) {
            const float4 weight = read_imagef(filter, sampler,
                                              (int2)(j + fwidth / 2,
                                              i + fheight / 2));
            const float4 iv = read_imagef(src, sampler, 
                                          coord + (int2)(j, i));
            e += iv.x * weight.x; 
        }
    }
    write_imagef(out, coord, (e / (float)(fwidth * fheight)));
}                              
#else                               
__kernel void filter_image(read_only image2d_t src,
                           read_only image2d_t filter,
                           __global real_t* out) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const int width = get_image_width(src);
    const int fwidth = get_image_width(filter);
    const int fheight = get_image_height(filter);
    coord += (int2)(fwidth / 2, fheight / 2);
    float e = 0.0f;
    for(int i = -fheight / 2; i <= fheight / 2; ++i) {
        for(int j = -fwidth / 2; j <= fwidth / 2; ++j) {
            const float4 weight = read_imagef(filter, sampler,
                                              (int2)(j + fwidth / 2,
                                              i + fheight / 2));
            const float4 iv = read_imagef(src, sampler, 
                                          coord + (int2)(j, i));
            e += iv.x * weight.x; 
        }
    }
    out[coord.y * width + coord.x] = e / (float)(fwidth * fheight);
}
#endif