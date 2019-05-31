/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "../../common/polybenchUtilFuncts.h"

#define BENCHMARK_NAME "GESUMMV"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 16384  // 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
             DATA_TYPE *tmp) {
    int i, j;

    for (i = 0; i < N; i++) {
        tmp[i] = 0;
        y[i] = 0;
        for (j = 0; j < N; j++) {
            tmp[i] = A[i * N + j] * x[j] + tmp[i];
            y[i] = B[i * N + j] * x[j] + y[i];
        }

        y[i] = ALPHA * tmp[i] + BETA * y[i];
    }
}

void gesummv_omp(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
                 DATA_TYPE *tmp) {
    int i, j;

#pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++) {
        tmp[i] = 0;
        y[i] = 0;
        for (j = 0; j < N; j++) {
            tmp[i] = A[i * N + j] * x[j] + tmp[i];
            y[i] = B[i * N + j] * x[j] + y[i];
        }

        y[i] = ALPHA * tmp[i] + BETA * y[i];
    }
}

void init(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *B) {
    int i, j;

    for (i = 0; i < N; i++) {
        x[i] = ((DATA_TYPE)i) / N;

        for (j = 0; j < N; j++) {
            A[i * N + j] = ((DATA_TYPE)i * j) / N;
            B[i * N + j] = ((DATA_TYPE)i * j) / N;
        }
    }
}

void compareResults(DATA_TYPE *y, DATA_TYPE *y_outputFromOmp) {
    int i, fail;
    fail = 0;

    for (i = 0; i < (N); i++) {
        if (percentDiff(y[i], y_outputFromOmp[i]) >
            PERCENT_DIFF_ERROR_THRESHOLD) {
            fail++;
        }
    }

    // Print results
    printf(
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
        "%d\n",
        PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, N);
    double t_start, t_end;

    // Need two sets of data to avoid data already loaded in cache
    DATA_TYPE *A;
    DATA_TYPE *B;
    DATA_TYPE *x;
    DATA_TYPE *y;
    DATA_TYPE *tmp;
    
    DATA_TYPE *A1;
    DATA_TYPE *B1;
    DATA_TYPE *x1;
    DATA_TYPE *y_outputFromOmp;
    DATA_TYPE *tmp1; 

    A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
    B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
    x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    A1 = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
    B1 = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
    x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    y_outputFromOmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    tmp1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

    init(A1, x1, B1);
    t_start = rtclock();
    gesummv_omp(A1, B1, x1, y_outputFromOmp, tmp1);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start);

    init(A, x, B);
    t_start = rtclock();
    gesummv(A, B, x, y, tmp);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(y, y_outputFromOmp);

    free(A);
    free(B);
    free(x);
    free(y);
    free(y_outputFromOmp);
    free(tmp);

    return 0;
}
