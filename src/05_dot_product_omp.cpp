#ifdef _OPENMP
#include <omp.h>
#endif
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

#ifdef _OPENMP
real_t dot(int N, const real_t* X, const real_t* Y) {
    real_t d = real_t(0);
    const real_t* x = 0;
    const real_t* y = 0;
    int n, i, nthr, mythr;
    #pragma omp parallel default(none) reduction(+:d) \
            shared(N, X, Y) private(n, i, nthr, mythr, x, y)
    {
        nthr = omp_get_num_threads();  
        mythr = omp_get_thread_num();
        n = N / nthr;
        x = X + n * mythr;
        y = Y + n * mythr;
        if(mythr == nthr - 1)
            n += N - n * nthr;
        d = x[0] * y[0];
        for(i = 1; i != n; ++i)
            d += x[i] * y[i];  

    }
    return d;
}
#else
real_t dot(int N, const real_t* X, const real_t* Y) {
    return std::inner_product(X, X + N, Y, real_t(0));
}
#endif 

//------------------------------------------------------------------------------
int main (int, char**) {
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
  assert(dot(n, a, b) == result);
  clock_gettime(CLOCK_MONOTONIC, &e);
  std::cout << "Time: " << time_diff_ms(s, e) << "ms" << std::endl;
  return 0;
}