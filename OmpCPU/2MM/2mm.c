/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define BENCHMARK_NAME "2MM"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define SIZE 2048

/* Problem size. */
#define NI SIZE
#define NJ SIZE
#define NK SIZE
#define NL SIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D) {
    int i, j;

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NK; j++) {
            A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < NK; i++) {
        for (j = 0; j < NJ; j++) {
            B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
        }
    }

    for (i = 0; i < NL; i++) {
        for (j = 0; j < NJ; j++) {
            C[i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
        }
    }

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NL; j++) {
            D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
        }
    }
}

void compareResults(DATA_TYPE* E, DATA_TYPE* E_outputFromOMP) {
    int i, j, fail;
    fail = 0;

    for (i = 0; i < NL; i++) {
        for (j = 0; j < NI; j++) {
            if (percentDiff(E[i * NI + j], E_outputFromOMP[i * NI + j]) >
                PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }

    // print results
    printf(
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
        "%d\n",
        PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D,
             DATA_TYPE* E) {
    int i, j, k;

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            C[i * NJ + j] = 0.0;
            for (k = 0; k < NK; ++k) {
                C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NL; j++) {
            E[i * NL + j] = 0.0;
            for (k = 0; k < NJ; ++k) {
                E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
            }
        }
    }
}

void mm2_omp(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D,
             DATA_TYPE* E) {
    int i, j, k;

#pragma omp parallel for private(i, j, k) collapse(2) 
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            C[i * NJ + j] = 0.0;
            for (k = 0; k < NK; ++k) {
                C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }

#pragma omp parallel for private(i, j, k) collapse(2)
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NL; j++) {
            E[i * NL + j] = 0.0;
            for (k = 0; k < NJ; ++k) {
                E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, NI);
    double t_start, t_end;

    DATA_TYPE* C;
    DATA_TYPE* A;
    DATA_TYPE* B;
    DATA_TYPE* D;
    DATA_TYPE* E;
    DATA_TYPE* E_outputFromOMP;

    C = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));
    A = (DATA_TYPE*)malloc(NI * NK * sizeof(DATA_TYPE));
    B = (DATA_TYPE*)malloc(NK * NJ * sizeof(DATA_TYPE));
    D = (DATA_TYPE*)malloc(NJ * NL * sizeof(DATA_TYPE));
    E = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));
    E_outputFromOMP = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));

    init_array(A, B, C, D);

    t_start = rtclock();
    mm2_omp(A, B, C, D, E_outputFromOMP);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start);

    t_start = rtclock();
    mm2_cpu(A, B, C, D, E);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(E, E_outputFromOMP);

    free(C);
    free(A);
    free(B);
    free(D);
    free(E);
    free(E_outputFromOMP);

    return 0;
}
