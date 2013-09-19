//Author: Ugo Varetto
//dot product with C++11: faster than OpenCL! and OpenMP on SandyBridge Xeons
//with g++4.8.1 -lrt -std=c++
//-DBLOCK enables block dot version (slower!),
//-DUSE_AVX enables block+avx(fastest)
//enable avx as needed by adding -mavx2 when -DUSE_AVX defined
//launch with: 
//a.out 268435456 64 (256 Mi doubles, 64 threads!) non-avx version
//a.out 268435456 32 (256 Mi doubles, 32 threads!) avx version

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
#include <ctime>
#include <numeric>
#include <cassert>
#include <vector>
#include <cstring>

typedef double real_t;
const double EPS = 1E-10;

//------------------------------------------------------------------------------
real_t time_diff_ms(const timespec& start, const timespec& end) {
    return end.tv_sec * 1E3 +  end.tv_nsec / 1E6
           - (start.tv_sec * 1E3 + start.tv_nsec / 1E6);  
}

#ifndef USE_AVX
//------------------------------------------------------------------------------
real_t dotblock(int N, const real_t* x, const real_t* y, int block) {
    std::vector< real_t > b1(block);
    std::vector< real_t > b2(block);
    real_t d = real_t(0);
    for(int b = 0; b < N; b += block) {
        std::copy(x + b, x + b + block, b1.begin());
        std::copy(y + b, y + b + block, b2.begin());   
        const std::ptrdiff_t diff = &b2[0] - &b1[0];
        for(real_t* p = &b1[0]; p != &b1[0] + b1.size(); ++p) {
            d += *p * (*(p + diff));
        }
    }
    return d;
}
#else
#ifndef BLOCK
#define BLOCK //AVX only supported through dotblock function
#endif
//------------------------------------------------------------------------------
#if 0
real_t dotblock(int sN, const real_t* x, const real_t* y, int sblock) {
    const int N = sN / 4;
    const int block = sblock / 4;
    __m256d* b1 = (__m256d*)_mm_malloc(sN * sizeof(double), 32);
    __m256d* b2 = (__m256d*)_mm_malloc(sN * sizeof(double), 32);
    __m256d d = {0, 0, 0, 0};
    for(int b = 0; b < N; b += block) {
       
        memcpy((__m256d*)b1, x + b, block * sizeof(double));
        memcpy((__m256d*)b2, y + b, block * sizeof(double)); 
       
        for(int i = 0; i != block; ++i) {
            d = _mm256_add_pd(_mm256_mul_pd(b1[i], b2[i]), d);
        }
    }
    _mm_free(b1);
    _mm_free(b2);
    return d[0] + d[1] + d[2] + d[3]; 
}
#endif
//------------------------------------------------------------------------------
std::function< real_t () > 
make_dotblock(int sN, const real_t* x, const real_t* y, int sblock) {
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
    __m256d* b1 = (__m256d*)_mm_malloc(2 * sN * sizeof(double), 32);
    __m256d* b2 = (__m256d*)_mm_malloc(2 * sN * sizeof(double), 32);
    return [=](bool cleanup = false) { //call with true to release resources
                                       //stored in closure
          if(cleanup) {
            _mm_free(b1);
            _mm_free(b2);
            return real_t(0);
          }
          __m256d d = {0, 0, 0, 0};
          for(int b = 0; b < N; b += block) {
              const int bsize = N - b < 2 * block ? N - b : block;
              memcpy((__m256d*)b1, x + b, bsize * sizeof(double));
              memcpy((__m256d*)b2, y + b, bsize * sizeof(double));
              for(int i = 0; i != bsize; ++i) {
                  d = _mm256_add_pd(_mm256_mul_pd(b1[i], b2[i]), d);
              }
          }
          return d[0] + d[1] + d[2] + d[3];
      }; 
}
#endif
//------------------------------------------------------------------------------
real_t dot(int N, const real_t* X, const real_t* Y, int nt) {
    std::vector< std::future< real_t > > futures;
    for(int i = 0; i != nt; ++i) {
        const int off = i * ( N / nt );
        const int size = i == nt - 1 ? N / nt + N % nt : N / nt;
        futures.push_back(
            std::async(std::launch::async,
#ifdef BLOCK                         
                       make_dotblock(size, X + off, Y + off, 16384)));
#else           
                       [X, Y, off] {     
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
                << " <size> <number of threads>" << std::endl;
      return 0;
  }
  const int N = atoi(argv[1]);//e.g. 1024 * 1024 * 256;
#ifdef BLOCK
  if(N % 4 != 0) {
      std::cout << "Invalid input:\n"
                "\tsize must be evenly divisible by 4 When compiled"
                "with -DBLOCK"
                << std::endl;
      return 0;
  }
#endif            
  std::vector< real_t > a(N);
  std::vector< real_t > b(N);
  std::default_random_engine rng(std::random_device{}()); 
  std::uniform_real_distribution< real_t > dist(1, 2);
  std::generate(a.begin(), a.end(), [&dist, &rng]{return dist(rng);});
  std::generate(b.begin(), b.end(), [&dist, &rng]{return dist(rng);});
  //result falls in [256Mi, 4 x 256Mi]
  const real_t result = 
    std::inner_product(a.begin(), a.end(), b.begin(), real_t(0));
  timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
  const real_t dotres = dot(N, &a[0], &b[0], atoi(argv[2]));
  clock_gettime(CLOCK_MONOTONIC, &e);
  if(std::abs(dotres - result > EPS))
      std::cerr << "ERROR: " << "got " << dotres << " instead of " 
                << result << " difference = " << (dotres - result) << std::endl;
  else
      std::cout << "PASSED" << std::endl;
  std::cout << "Time: " << time_diff_ms(s, e) << "ms" << std::endl;
  return 0;
}