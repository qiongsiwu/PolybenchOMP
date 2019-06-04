/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define BENCHMARK_NAME "MVT"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 16384

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* OpenMP num teams and team sizes */
#define NUM_TEAMS 512
#define TEAM_SIZE 128 

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1,
                DATA_TYPE* y2) {
    int i, j;

    for (i = 0; i < N; i++) {
        x1[i] = ((DATA_TYPE)i) / N;
        x2[i] = ((DATA_TYPE)i + 1) / N;
        y1[i] = ((DATA_TYPE)i + 3) / N;
        y2[i] = ((DATA_TYPE)i + 4) / N;
        for (j = 0; j < N; j++) {
            A[i * N + j] = ((DATA_TYPE)i * j) / N;
        }
    }
}

void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1,
            DATA_TYPE* y2) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            x1[i] = x1[i] + a[i * N + j] * y1[j];
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            x2[i] = x2[i] + a[j * N + i] * y2[j];
        }
    }
}

void runMvt_omp(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1,
                DATA_TYPE* y2) {
    int i, j;

#pragma omp target data map(to: a[0 : N * N], y1[0: N], y2[0 : N]) \
                        map(tofrom: x1[0 : N], x2[0 : N])
    {
#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for private(i, j)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                x1[i] = x1[i] + a[i * N + j] * y1[j];
            }
        }

#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for private(i, j)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                x2[i] = x2[i] + a[j * N + i] * y2[j];
            }
        }
    }
}

void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromOmp, DATA_TYPE* x2,
                    DATA_TYPE* x2_outputFromOmp) {
    int i, fail;
    fail = 0;

    for (i = 0; i < N; i++) {
        if (percentDiff(x1[i], x1_outputFromOmp[i]) >
            PERCENT_DIFF_ERROR_THRESHOLD) {
            fail++;
        }

        if (percentDiff(x2[i], x2_outputFromOmp[i]) >
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

int main() {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, N);
    double t_start, t_end;

    DATA_TYPE* a;
    DATA_TYPE* a_omp;
    DATA_TYPE* x1;
    DATA_TYPE* x2;
    DATA_TYPE* x1_outputFromOmp;
    DATA_TYPE* x2_outputFromOmp;
    DATA_TYPE* y_1;
    DATA_TYPE* y_2;
    DATA_TYPE* y_1_omp;
    DATA_TYPE* y_2_omp;

    a = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
    a_omp = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
    x1 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    x2 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    x1_outputFromOmp = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    x2_outputFromOmp = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    y_1 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    y_2 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    y_1_omp = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
    y_2_omp = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));

    init_array(a_omp, x1_outputFromOmp, x2_outputFromOmp, y_1_omp, y_2_omp);
    t_start = rtclock();
    runMvt_omp(a_omp, x1_outputFromOmp, x2_outputFromOmp, y_1_omp, y_2_omp);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST
    init_array(a, x1, x2, y_1, y_2); 
    t_start = rtclock();
    runMvt(a, x1, x2, y_1, y_2);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(x1, x1_outputFromOmp, x2, x2_outputFromOmp);
#endif

    free(a);
    free(x1);
    free(x2);
    free(x1_outputFromOmp);
    free(x2_outputFromOmp);
    free(y_1);
    free(y_2);

    return 0;
}
