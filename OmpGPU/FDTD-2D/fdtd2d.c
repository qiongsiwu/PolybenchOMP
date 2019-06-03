/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define BENCHMARK_NAME "FDTD-2D"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
#define tmax 500
#define NX 2048
#define NY 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* OpenMP number of teams and team sizes */
#define NUM_TEAMS 256 
#define TEAM_SIZE 256 

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey,
                 DATA_TYPE* hz) {
    int i, j;

    for (i = 0; i < tmax; i++) {
        _fict_[i] = (DATA_TYPE)i;
    }

    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
    }
}

void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz) {
    int t, i, j;

    for (t = 0; t < tmax; t++) {
        for (j = 0; j < NY; j++) {
            ey[0 * NY + j] = _fict_[t];
        }

        for (i = 1; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                ey[i * NY + j] = ey[i * NY + j] -
                                 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }

        for (i = 0; i < NX; i++) {
            for (j = 1; j < NY; j++) {
                ex[i * (NY + 1) + j] =
                    ex[i * (NY + 1) + j] -
                    0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }

        for (i = 0; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                hz[i * NY + j] =
                    hz[i * NY + j] -
                    0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                           ey[(i + 1) * NY + j] - ey[i * NY + j]);
            }
        }
    }
}

void runFdtd_omp(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz) {
    int t, i, j;

#pragma omp target data map(to: _fict_[0:tmax]) \
                        map(tofrom: ex[0: NX * (NY + 1)], ey[0: (NX + 1) * NY]) \
                        map(tofrom: hz[0 : NX * NY])
    for (t = 0; t < tmax; t++) {
#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for private(j)
        for (j = 0; j < NY; j++) {
            ey[0 * NY + j] = _fict_[t];
        }

#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for private(i, j) collapse(2)
        for (i = 1; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                ey[i * NY + j] = ey[i * NY + j] -
                                 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }

#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for private(i, j) collapse(2)
        for (i = 0; i < NX; i++) {
            for (j = 1; j < NY; j++) {
                ex[i * (NY + 1) + j] =
                    ex[i * (NY + 1) + j] -
                    0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }

#pragma omp target teams num_teams(NUM_TEAMS) thread_limit(TEAM_SIZE)
#pragma omp distribute parallel for private(i, j) collapse(2)
        for (i = 0; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                hz[i * NY + j] =
                    hz[i * NY + j] -
                    0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                           ey[(i + 1) * NY + j] - ey[i * NY + j]);
            }
        }
    }
}

void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2) {
    int i, j, fail;
    fail = 0;

    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            if (percentDiff(hz1[i * NY + j], hz2[i * NY + j]) >
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

int main() {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, NX);
    double t_start, t_end;

    DATA_TYPE* _fict_;
    DATA_TYPE* ex;
    DATA_TYPE* ey;
    DATA_TYPE* hz;
    DATA_TYPE* hz_outputFromOmp;

    _fict_ = (DATA_TYPE*)malloc(tmax * sizeof(DATA_TYPE));
    ex = (DATA_TYPE*)malloc(NX * (NY + 1) * sizeof(DATA_TYPE));
    ey = (DATA_TYPE*)malloc((NX + 1) * NY * sizeof(DATA_TYPE));
    hz = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));
    hz_outputFromOmp = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));

    init_arrays(_fict_, ex, ey, hz_outputFromOmp);
    t_start = rtclock();
    runFdtd_omp(_fict_, ex, ey, hz_outputFromOmp);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start);

    init_arrays(_fict_, ex, ey, hz);
    t_start = rtclock();
    runFdtd(_fict_, ex, ey, hz);
    t_end = rtclock();

    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(hz, hz_outputFromOmp);

    free(_fict_);
    free(ex);
    free(ey);
    free(hz);
    free(hz_outputFromOmp);

    return 0;
}
