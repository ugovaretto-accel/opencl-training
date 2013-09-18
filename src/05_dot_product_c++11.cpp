//Author: Ugo Varetto
//dot product with C++11: faster than OpenCL! and OpenMP on SandyBridge Xeons
//with g++4.8.1 -lrt -std=c++
//a.out 268435456 64 (256 Mi doubles, 64 threads!)

#if __cplusplus < 201103L
#error "C++ 11 required"
#endif

#include <thread>
#include <future>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <numeric>
#include <cassert>
#include <vector>

typedef double real_t;
const double EPS = 0.00000001;

//------------------------------------------------------------------------------
real_t time_diff_ms(const timespec& start, const timespec& end) {
    return end.tv_sec * 1E3 +  end.tv_nsec / 1E6
           - (start.tv_sec * 1E3 + start.tv_nsec / 1E6);  
}

//------------------------------------------------------------------------------
real_t dotblock(const real_t* x, const real_t* y, int N, int block) {
    std::vector< real_t > b1(block);
    std::vector< real_t > b2(block);
    real_t d = real_t(0);
    for(int b = 0; b < N; b += block) {
        std::copy(x + b, x + b + block, b1.begin());
        std::copy(y + b, y + b + block, b2.begin());
        d += std::inner_product(b1.begin(), b1.end(), b2.begin(), real_t(0));
    }
    return d;
}

//------------------------------------------------------------------------------
real_t dot(int N, const real_t* X, const real_t* Y, int nt) {
    std::vector< std::future< real_t > > futures;
    for(int i = 0; i < (nt - 1); ++i) {
        futures.push_back(
            std::async(std::launch::async, [nt, X, Y, N](int off) {
                //return dotblock(X + off, Y + off, N / nt, 16384); 
                return std::inner_product(X + off, X + off + N / nt,
                                          Y + off, real_t(0));  
        }, i * (N / nt)));  
    }
    futures.push_back(std::async([nt, X, Y, N]() {
      const int n =  N / nt + N % nt; 
      const int off = (nt-1) * (N / nt);
      return std::inner_product(X + off, X + off + n, Y + off, real_t(0));
      //return dotblock(X + off, Y + off, N / nt, 16384);  
    }));
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
  std::vector< real_t > a(N);
  std::vector< real_t > b(N);
  std::default_random_engine rng(std::random_device{}()); 
  std::uniform_real_distribution< real_t > dist(1, 2);
  std::generate(a.begin(), a.end(), [&dist, &rng]{return dist(rng);});
  std::generate(b.begin(), b.end(), [&dist, &rng]{return dist(rng);});
  const real_t result = 
    std::inner_product(a.begin(), a.end(), b.begin(), real_t(0));
  timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
  const real_t dotres = dot(N, &a[0], &b[0], atoi(argv[2]));

  if(std::abs(dotres - result > EPS))
      std::cerr << "ERROR: " << "got " << dotres << " instead of " 
                << result << " difference = " << (dotres - result) << std::endl;
  else
      std::cout << "PASSED" << std::endl;
  clock_gettime(CLOCK_MONOTONIC, &e);
  std::cout << "Time: " << time_diff_ms(s, e) << "ms" << std::endl;
  return 0;
}