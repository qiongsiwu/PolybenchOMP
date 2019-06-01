/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <omp.h> 

#include "../../common/polybenchUtilFuncts.h"

#define BENCHMARK_NAME "2DCONV"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define SIZE 16384

#define NI SIZE
#define NJ SIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Number of teams and team sizes */
#define NUM_TEAMS 128
#define BLOCK_SIZE 256

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv2D(DATA_TYPE *A, DATA_TYPE *B) {
    int i, j;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;
    c21 = +0.5;
    c31 = -0.8;
    c12 = -0.3;
    c22 = +0.6;
    c32 = -0.9;
    c13 = +0.4;
    c23 = +0.7;
    c33 = +0.10;

    for (i = 1; i < NI - 1; ++i) // 0
    {
        for (j = 1; j < NJ - 1; ++j) // 1
        {
            B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] +
                            c12 * A[(i + 0) * NJ + (j - 1)] +
                            c13 * A[(i + 1) * NJ + (j - 1)] +
                            c21 * A[(i - 1) * NJ + (j + 0)] +
                            c22 * A[(i + 0) * NJ + (j + 0)] +
                            c23 * A[(i + 1) * NJ + (j + 0)] +
                            c31 * A[(i - 1) * NJ + (j + 1)] +
                            c32 * A[(i + 0) * NJ + (j + 1)] +
                            c33 * A[(i + 1) * NJ + (j + 1)];
        }
    }
}

void conv2D_omp(DATA_TYPE *A, DATA_TYPE *B) {
    int i, j;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;
    c21 = +0.5;
    c31 = -0.8;
    c12 = -0.3;
    c22 = +0.6;
    c32 = -0.9;
    c13 = +0.4;
    c23 = +0.7;
    c33 = +0.10;

#pragma omp target map(to : A [0:NI * NJ]) map(from : B [0:NI * NJ])
#pragma omp teams num_teams(NUM_TEAMS) thread_limit(BLOCK_SIZE) 
#pragma omp distribute parallel for private(i, j)
    for (i = 1; i < NI - 1; ++i) // 0
    {
        /*if (omp_get_team_num() == 0 && omp_get_thread_num() == 0){
            printf("Num teams: %d\n", omp_get_num_teams()); 
            printf("Num threads: %d\n", omp_get_num_threads()); 
        }*/

        for (j = 1; j < NJ - 1; ++j) // 1
        {
            B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] +
                            c12 * A[(i + 0) * NJ + (j - 1)] +
                            c13 * A[(i + 1) * NJ + (j - 1)] +
                            c21 * A[(i - 1) * NJ + (j + 0)] +
                            c22 * A[(i + 0) * NJ + (j + 0)] +
                            c23 * A[(i + 1) * NJ + (j + 0)] +
                            c31 * A[(i - 1) * NJ + (j + 1)] +
                            c32 * A[(i + 0) * NJ + (j + 1)] +
                            c33 * A[(i + 1) * NJ + (j + 1)];
        }
    }
}

void init(DATA_TYPE *A) {
    int i, j;

    for (i = 0; i < NI; ++i) {
        for (j = 0; j < NJ; ++j) {
            A[i * NJ + j] = (float)rand() / RAND_MAX;
        }
    }
}

void compareResults(DATA_TYPE *B, DATA_TYPE *B_outputFromOMP) {
    int i, j, fail;
    fail = 0;

    // Compare a and b
    for (i = 1; i < (NI - 1); i++) {
        for (j = 1; j < (NJ - 1); j++) {
            if (percentDiff(B[i * NJ + j], B_outputFromOMP[i * NJ + j]) >
                PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }

    // Print results
    printf(
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
        "%d\n",
        PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, NI);
    double t_start, t_end;

    DATA_TYPE *A;
    DATA_TYPE *B;
    DATA_TYPE *B_outputFromOMP;

    A = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
    B = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
    B_outputFromOMP = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

    init(A);

    t_start = rtclock();
    conv2D_omp(A, B_outputFromOMP);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start); //);

    t_start = rtclock();
    conv2D(A, B);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); //);
    compareResults(B, B_outputFromOMP);

    free(A);
    free(B);
    free(B_outputFromOMP);

    return 0;
}
