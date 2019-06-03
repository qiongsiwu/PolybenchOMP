/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define GPU_DEVICE 0

#define BENCHMARK_NAME "3MM"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define SIZE 512

/* Problem size. */
#define NI SIZE
#define NJ SIZE
#define NK SIZE
#define NL SIZE
#define NM SIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Number of teams and team sizes */
#define NUM_TEAMS 128
#define TEAM_SIZE 128

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D) {
    int i, j;

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NK; j++) {
            A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < NK; i++) {
        for (j = 0; j < NJ; j++) {
            B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
        }
    }

    for (i = 0; i < NJ; i++) {
        for (j = 0; j < NM; j++) {
            C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
        }
    }

    for (i = 0; i < NM; i++) {
        for (j = 0; j < NL; j++) {
            D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
        }
    }
}

void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromOmp) {
    int i, j, fail;
    fail = 0;

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NL; j++) {
            if (percentDiff(G[i * NL + j], G_outputFromOmp[i * NL + j]) >
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

void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G) {
    int i, j, k;

    /* E := A*B */
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            E[i * NJ + j] = 0;
            for (k = 0; k < NK; ++k) {
                E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }

    /* F := C*D */
    for (i = 0; i < NJ; i++) {
        for (j = 0; j < NL; j++) {
            F[i * NL + j] = 0;
            for (k = 0; k < NM; ++k) {
                F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
            }
        }
    }

    /* G := E*F */
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NL; j++) {
            G[i * NL + j] = 0;
            for (k = 0; k < NJ; ++k) {
                G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
            }
        }
    }
}

void mm3_omp(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G) {

    int i, j, k;

    /* E := A*B */
#pragma omp target map(to                                                      \
                       : A [0:NI * NK], B [0:NK * NJ]) map(from                \
                                                           : E [0:NI * NJ])
#pragma omp teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for collapse(2) private(i, j, k)
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            E[i * NJ + j] = 0;
            for (k = 0; k < NK; ++k) {
                E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }

    /* F := C*D */
#pragma omp target map(to                                                      \
                       : C [0:NJ * NM], D [0:NM * NL]) map(from                \
                                                           : F [0:NJ * NL])
#pragma omp teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for collapse(2) private(i, j, k)
    for (i = 0; i < NJ; i++) {
        for (j = 0; j < NL; j++) {
            F[i * NL + j] = 0;
            for (k = 0; k < NM; ++k) {
                F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
            }
        }
    }

    /* G := E*F */
#pragma omp target map(tofrom                                                  \
                       : E [0:NI * NJ], F [0:NJ * NL]) map(from                \
                                                           : G [0:NI * NL])
#pragma omp teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for collapse(2) private(i, j, k)
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NL; j++) {
            G[i * NL + j] = 0;
            for (k = 0; k < NJ; ++k) {
                G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, NI);
    double t_start, t_end;

    DATA_TYPE *A;
    DATA_TYPE *B;
    DATA_TYPE *C;
    DATA_TYPE *D;
    DATA_TYPE *E;
    DATA_TYPE *F;
    DATA_TYPE *G;
    DATA_TYPE *G_outputFromOmp;

    A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
    B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
    C = (DATA_TYPE *)malloc(NJ * NM * sizeof(DATA_TYPE));
    D = (DATA_TYPE *)malloc(NM * NL * sizeof(DATA_TYPE));
    E = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
    F = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
    G = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));
    G_outputFromOmp = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));

    init_array(A, B, C, D);

    t_start = rtclock();
    mm3_omp(A, B, C, D, E, F, G_outputFromOmp);
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST
    t_start = rtclock();
    mm3_cpu(A, B, C, D, E, F, G);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(G, G_outputFromOmp);
#endif

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
    free(G);
    free(G_outputFromOmp);

    return 0;
}
