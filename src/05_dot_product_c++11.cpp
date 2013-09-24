//Author: Ugo Varetto
//dot product with C++11: faster than OpenCL! on SandyBridge Xeons
//with g++4.8.1 -std=c++ -O3 -pthread
//-DBLOCK enables block dot version (slower!),
//-DUSE_AVX enables block+avx(fastest)
//enable avx as needed by adding -mavx2 when -DUSE_AVX defined
//launch with: 
//a.out 268435456 64 (256 Mi doubles, 64 threads!) non-avx version
//a.out 268435456 16 (256 Mi doubles, 32 threads!) avx version
//Note: with 256Mi doubles the avx code is also faster than the CUDA
//version running on a K20x

#if __cplusplus < 201103L
#error "C++ 11 required"
#endif
#ifdef USE_AVX
#include <immintrin.h>
#include <mm_malloc.h>
#endif
#include <thread>
#include <future>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>
#include <cassert>
#include <vector>
#include <cstring> //memcpy
#include <exception>

typedef double real_t;
const double EPS = 1E-10; //consider making this a relative error dependent
                          //on the size of the input; it might happen
                          //that you get errors in the order of 10-5 with
                          //256Mi positive elements

//------------------------------------------------------------------------------
real_t time_diff_ms(
    const std::chrono::time_point< std::chrono::steady_clock >& s,
    const std::chrono::time_point< std::chrono::steady_clock >& e) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count();  
}

#ifndef USE_AVX
//------------------------------------------------------------------------------
std::function< real_t () > 
make_dotblock(int N, const real_t* x, const real_t* y, int block) {
    //in case the size is not evenly divisible by the block size
    //we need to allocate additional bytes in the buffers in order
    //to copy block + N % block elements
    //
    //Problem: if you declare the std::vector outside the lambda function
    //and pass it by value to each closure through [=], declaring
    //the lambda 'mutable', it is slower than declaring it inside the 
    //body of the lambda; if you try to pass it by refreence
    //through [&] you get a segfault
    //Solution: declare empty vectors and issue a resize inside the lambda
    //function: this does not make any difference in case the lamda is called
    //only once as in this case, but when calling it multiple times with the
    //same data size it does speed up operations because no reallocation is
    //performed 
    std::vector< real_t > b1(0);
    std::vector< real_t > b2(0);
    return [=]() mutable { 
        b1.resize(2 * N);
        b2.resize(2 * N); 
        real_t d = real_t(0);
        for(int b = 0; b < N; b += block) {
             const int bsize = N - b < 2 * block ? N - b : block;
             std::copy(x + b, x + b + bsize, b1.begin());
             std::copy(y + b, y + b + bsize, b2.begin());
             for(int i = 0; i != bsize; ++i) {
                 d += b1[i] * b2[i];
             }
         }
         return d;
      }; 
}
#else
#ifndef BLOCK
#define BLOCK //AVX only supported through dotblock function
#endif
//------------------------------------------------------------------------------
std::function< real_t () > 
make_dotblock_avx(int sN, const real_t* x, const real_t* y, int sblock) {
    //requirement: multiple of 4 doubles
    assert(sN % 4 == 0);
    assert(sblock % 4  == 0);
    const int N = sN / 4;
    const int block = sblock / 4;
    //memory is allocated outside of the lambda function and released
    //when the function is called with cleanup == true
    //in case the size is not evenly divisible by the block size
    //we need to allocate additional bytes in the buffers in order
    //to copy block + N % block elements
    __m256d* b1 = (__m256d*)_mm_malloc(2 * sN * sizeof(real_t), 32);
    __m256d* b2 = (__m256d*)_mm_malloc(2 * sN * sizeof(real_t), 32);
    //note that although we are not declaring the lambda 'mutable' we can
    //still copy into the buffers since all we need is a pointer to memory
    //passed by value
    return [=](bool cleanup = true) { //call with true to release resources
                                      //stored in closure;
                                      //disable cleanup when lambda called
                                      //multiple times with the same data size
        __m256d d = {0, 0, 0, 0};
        for(int b = 0; b < N; b += block) {
            const int bsize = N - b < 2 * block ? N - b : block;
            memcpy((__m256d*)b1, x + b, bsize * sizeof(real_t));
            memcpy((__m256d*)b2, y + b, bsize * sizeof(real_t));
            for(int i = 0; i != bsize; ++i) {
                d = _mm256_add_pd(_mm256_mul_pd(b1[i], b2[i]), d);
            }
        }
        if(cleanup) {
          _mm_free(b1);
          _mm_free(b2);
          return real_t(0);
        }
        return d[0] + d[1] + d[2] + d[3];
    }; 
}
#endif
//------------------------------------------------------------------------------
real_t dot(int N, const real_t* X, const real_t* Y, int nt,
           int blocksize = 16384) {
    std::vector< std::future< real_t > > futures;
    for(int i = 0; i != nt; ++i) {
        const int off = i * ( N / nt );
        const int size = i == nt - 1 ? N / nt + N % nt : N / nt;
        futures.push_back(
            std::async(std::launch::async,
#ifdef BLOCK 
#ifdef USE_AVX                       
                       make_dotblock_avx(size, X + off, Y + off, blocksize)));
#else
                       make_dotblock(size, X + off, Y + off, blocksize)));
#endif        
#else           
                       [X, Y, off, size]() {     
                           return std::inner_product(X + off, X + off + size,
                                                     Y + off, real_t(0));
                       }));                                 
#endif
    }                
    real_t d = real_t(0); 
    std::for_each(futures.begin(), futures.end(),
                    [&d](std::future< real_t >& f) {                   
                        d += f.get();  
                    });
    return d;
}
//------------------------------------------------------------------------------
int main (int argc, char** argv) {

  if(argc < 3 || atoi(argv[1]) < 1 || atoi(argv[2]) < 1) {
      std::cout << "usage: " << argv[0] 
                << " <size> <number of threads>"
#ifndef BLOCK     
                << " [block size, default = 16384]"               
#endif
                << std::endl;
      return 0;
  }
  
  std::cout << std::thread::hardware_concurrency() 
            << " concurrent threads are supported.\n\n";

  const int N = atoi(argv[1]);//e.g. 1024 * 1024 * 256;
  int blocksize = 16384;
  if(argc > 3) blocksize = atoi(argv[3]);
#ifdef USE_AVX
  if(sizeof(real_t) != sizeof(double)) {
      std::cout << "real_t must be declared as 'double' when "
                   "-DUSE_AVX specified" << std::endl;
      return 0;               
  }
  if(N % 4 != 0) {
      std::cout << "Invalid input:\n"
                "\tsize must be evenly divisible by 4 When compiled "
                "with -DUSE_AVX"
                << std::endl;
      return 0;
  }
#endif
  try {            
      std::vector< real_t > a(N);
      std::vector< real_t > b(N);
      std::default_random_engine rng(std::random_device{}()); 
      std::uniform_real_distribution< real_t > dist(1, 2);
      std::generate(a.begin(), a.end(), [&dist, &rng]{return dist(rng);});
      std::generate(b.begin(), b.end(), [&dist, &rng]{return dist(rng);});
      //result falls in [256Mi, 4 x 256Mi]
      const real_t result = 
        std::inner_product(a.begin(), a.end(), b.begin(), real_t(0));
      std::chrono::time_point< std::chrono::steady_clock > s, e;
      s = std::chrono::steady_clock::now();
      const real_t dotres = dot(N, &a[0], &b[0], atoi(argv[2]), blocksize);
      e = std::chrono::steady_clock::now();
      if(std::abs(dotres - result > EPS))
          std::cerr << "ERROR: " << "got " << dotres << " instead of " 
                    << result << " difference = " << (dotres - result) 
                    << std::endl;
      else
          std::cout << "PASSED" << std::endl;
      std::cout << "Time: " << time_diff_ms(s, e) << "ms" << std::endl;
  } catch(const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << std::endl;
      return EXIT_FAILURE;  
  }    
  return 0;
}
