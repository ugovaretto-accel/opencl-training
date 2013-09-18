#ifdef PARALLEL_
#if __cplusplus < 201103L
#error "C++ 11 required"
#endif
#endif

#include <thread>
#include <future>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <numeric>
#include <cassert>

typedef double real_t;

//------------------------------------------------------------------------------
real_t time_diff_ms(const timespec& start, const timespec& end) {
    return end.tv_sec * 1E3 +  end.tv_nsec / 1E6
           - (start.tv_sec * 1E3 + start.tv_nsec / 1E6);  
}

#ifdef PARALLEL_
real_t dot(int N, const real_t* X, const real_t* Y, int nt) {
    std::vector< std::future< real_t > > futures(nt);
    for(int i = 0; i != nt - 1; ++i) {
        futures.push_back(std::async([nt, X, Y, N](int i) {
            return std::inner_product(X + i * N / nt, X + i * N/ nt + N / nt, Y, real_t(0));  
        },i));  
    }
    futures.push_back(std::async([nt, X, Y, N]() {
      const int n =  N / nt + N % nt; 
      return std::inner_product(X + (nt-1) * N / nt, X + (nt-1) * N/ nt + n, Y, real_t(0));  
    }));
    real_t d = real_t(0); 
    std::for_each(futures.begin(), futures.end(),
                    [&d](std::future< real_t >& f) {
                        d += f.get();  
                    });
    return d;
}
#else
real_t dot(int N, const real_t* X, const real_t* Y) {
    return std::inner_product(X, X + N, Y, real_t(0));
}
#endif 

//------------------------------------------------------------------------------
int main (int, char** argv) {
  const int n = 1024*256; 
  double a[n], b[n];
  double result = 0;
  for(int i = 0; i < n; i++) {
      a[i] = i * 1.0;
      b[i] = i * 2.0;
      result += a[i] * b[i];
  }
  timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
#ifdef PARALLEL_  
  assert(dot(n, a, b, atoi(argv[1])) == result);
#else
  assert(dot(n, a, b) == result);
#endif
  clock_gettime(CLOCK_MONOTONIC, &e);
  std::cout << "Time: " << time_diff_ms(s, e) << "ms" << std::endl;
  return 0;
}