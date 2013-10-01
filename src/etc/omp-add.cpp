#include <omp.h>
#include <cstdio>
#define CHUNKSIZE 100
#define N     1000

main () {

int i, chunk;
float a[N], b[N], c[N];

/* Some initializations */
for (i=0; i < N; i++)
  a[i] = b[i] = i * 1.0;
chunk = CHUNKSIZE;
printf("%d", omp_get_max_threads());
#pragma omp parallel shared(a,b,c,chunk) private(i)
  {
   printf("%d", omp_get_max_threads());
  #pragma omp for schedule(dynamic,chunk) nowait
  for (i=0; i < N; i++) {
    printf("%d\n", omp_get_thread_num());
    c[i] = a[i] + b[i];
  }

  }  /* end of parallel section */

}