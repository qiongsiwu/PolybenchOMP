/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "../../common/polybenchUtilFuncts.h"

#define BENCHMARK_NAME "ATAX"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define SIZE 16384  // 4096

#define NX SIZE
#define NY SIZE

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A) {
    int i, j;

    for (i = 0; i < NX; i++) {
        x[i] = i * M_PI;
        for (j = 0; j < NY; j++) {
            A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
        }
    }
}

void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromOmp) {
    int i, fail;
    fail = 0;

    for (i = 0; i < NY; i++) {
        if (percentDiff(z[i], z_outputFromOmp[i]) >
            PERCENT_DIFF_ERROR_THRESHOLD) {
            fail++;
        }
    }

    // print results
    printf(
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
        "%d\n",
        PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void atax_cpu(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
    int i, j;

    for (i = 0; i < NY; i++) {
        y[i] = 0;
    }

    for (i = 0; i < NX; i++) {
        tmp[i] = 0;

        for (j = 0; j < NY; j++) {
            tmp[i] = tmp[i] + A[i * NY + j] * x[j];
        }

        for (j = 0; j < NY; j++) {
            y[j] = y[j] + A[i * NY + j] * tmp[i];
        }
    }
}

void atax_omp(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {

    for (int i = 0; i < NY; i++) {
        y[i] = 0;
    }

#pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        tmp[i] = 0; 
        for (int j = 0; j < NY; j++) {
            tmp[i] += A[i * NY + j] * x[j]; 
        }
    }

#pragma omp parallel for 
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            y[j] = y[j] + A[i * NY + j] * tmp[i];
        }
    }
}

int main(int argc, char **argv) {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, NX);
    double t_start, t_end;

    DATA_TYPE *A;
    DATA_TYPE *x;
    DATA_TYPE *y;
    DATA_TYPE *y_outputFromOmp;
    DATA_TYPE *tmp;

    A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
    x = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
    y = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
    y_outputFromOmp = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
    tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

    init_array(x, A);

    t_start = rtclock();
    atax_omp(A, x, y_outputFromOmp, tmp);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start);

    t_start = rtclock();
    atax_cpu(A, x, y, tmp);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(y, y_outputFromOmp);

    free(A);
    free(x);
    free(y);
    free(y_outputFromOmp);
    free(tmp);

    return 0;
}
