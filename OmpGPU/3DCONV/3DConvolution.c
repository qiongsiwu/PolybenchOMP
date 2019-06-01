/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define BENCHMARK_NAME "3DCONV"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#define SIZE 256

/* Problem size */
#define NI SIZE
#define NJ SIZE
#define NK SIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Number of teams and team sizes */
#define NUM_TEAMS 512 
#define TEAM_SIZE 256 

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv3D(DATA_TYPE* A, DATA_TYPE* B) {
    int i, j, k;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +2;
    c21 = +5;
    c31 = -8;
    c12 = -3;
    c22 = +6;
    c32 = -9;
    c13 = +4;
    c23 = +7;
    c33 = +10;

    for (i = 1; i < NI - 1; ++i)  // 0
    {
        for (j = 1; j < NJ - 1; ++j)  // 1
        {
            for (k = 1; k < NK - 1; ++k)  // 2
            {
                // printf("i:%d\nj:%d\nk:%d\n", i, j, k);
                B[i * (NK * NJ) + j * NK + k] =
                    c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
                    c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
                    c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
                    c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
                    c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
                    c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
                    c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
                    c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
                    c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
            }
        }
    }
}

void conv3D_omp(DATA_TYPE* A, DATA_TYPE* B) {
    int i, j, k;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +2;
    c21 = +5;
    c31 = -8;
    c12 = -3;
    c22 = +6;
    c32 = -9;
    c13 = +4;
    c23 = +7;
    c33 = +10;
#pragma omp target map(to: A[0 : NI * NJ * NK]) map(from: B[0 : NI * NJ * NK])
#pragma omp teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for private(i, j, k)  
    for (i = 1; i < NI - 1; ++i)  // 0
    {
        for (j = 1; j < NJ - 1; ++j)  // 1
        {
            for (k = 1; k < NK - 1; ++k)  // 2
            {
                // printf("i:%d\nj:%d\nk:%d\n", i, j, k);
                B[i * (NK * NJ) + j * NK + k] =
                    c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                    c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
                    c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
                    c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
                    c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
                    c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
                    c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
                    c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
                    c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
                    c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
            }
        }
    }
}

void init(DATA_TYPE* A) {
    int i, j, k;

    for (i = 0; i < NI; ++i) {
        for (j = 0; j < NJ; ++j) {
            for (k = 0; k < NK; ++k) {
                A[i * (NK * NJ) + j * NK + k] =
                   i % 12 + 2 * (j % 7) + 3 * (k % 13);
            }
        }
    }
}

void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromOMP) {
    int i, j, k, fail;
    fail = 0;

    // Compare result from cpu and gpu...
    for (i = 1; i < NI - 1; ++i)  // 0
    {
        for (j = 1; j < NJ - 1; ++j)  // 1
        {
            for (k = 1; k < NK - 1; ++k)  // 2
            {
                if (percentDiff(B[i * (NK * NJ) + j * NK + k],
                                B_outputFromOMP[i * (NK * NJ) + j * NK + k]) >
                    PERCENT_DIFF_ERROR_THRESHOLD) {
                    fail++;
                }
            }
        }
    }

    // Print results
    printf(
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
        "%d\n",
        PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


int main(int argc, char* argv[]) {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, NI);
    double t_start, t_end;

    DATA_TYPE* A;
    DATA_TYPE* B;
    DATA_TYPE* B_outputFromOMP;

    A = (DATA_TYPE*)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
    B = (DATA_TYPE*)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
    B_outputFromOMP = (DATA_TYPE*)malloc(NI * NJ * NK * sizeof(DATA_TYPE));

    init(A);

    t_start = rtclock();
    conv3D_omp(A, B_outputFromOMP);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST
    t_start = rtclock();
    conv3D(A, B);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(B, B_outputFromOMP);
#endif

    free(A);
    free(B);
    free(B_outputFromOMP);

    return 0;
}
