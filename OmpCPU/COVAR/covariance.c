/**
 * covariance.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define BENCHMARK_NAME "COVAR"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

/* Problem size */
#define SIZE 2048
#define M SIZE
#define N SIZE

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 32
#define DIM_THREAD_BLOCK_KERNEL_2_Y 8

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 256
#define DIM_THREAD_BLOCK_KERNEL_3_Y 1

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* data) {
    int i, j;

    for (i = 1; i < (M + 1); i++) {
        for (j = 1; j < (N + 1); j++) {
            data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
        }
    }
}

void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean) {
    int i, j, j1, j2;

    /* Determine mean of column vectors of input data matrix */
    for (j = 1; j < (M + 1); j++) {
        mean[j] = 0.0;
        for (i = 1; i < (N + 1); i++) {
            mean[j] += data[i * (M + 1) + j];
        }
        mean[j] /= FLOAT_N;
    }

    /* Center the column vectors. */
    for (i = 1; i < (N + 1); i++) {
        for (j = 1; j < (M + 1); j++) {
            data[i * (M + 1) + j] -= mean[j];
        }
    }

    /* Calculate the m * m covariance matrix. */
    for (j1 = 1; j1 < (M + 1); j1++) {
        for (j2 = j1; j2 < (M + 1); j2++) {
            symmat[j1 * (M + 1) + j2] = 0.0;
            for (i = 1; i < N + 1; i++) {
                symmat[j1 * (M + 1) + j2] +=
                    data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
            }
            symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
        }
    }
}

void covariance_omp(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean) {
    int i, j, j1, j2;

    /* Determine mean of column vectors of input data matrix */
#pragma omp parallel for private(j, i)
    for (j = 1; j < (M + 1); j++) {
        mean[j] = 0.0;
        for (i = 1; i < (N + 1); i++) {
            mean[j] += data[i * (M + 1) + j];
        }
        mean[j] /= FLOAT_N;
    }

    /* Center the column vectors. */
#pragma omp parallel for private(i, j) collapse(2)
    for (i = 1; i < (N + 1); i++) {
        for (j = 1; j < (M + 1); j++) {
            data[i * (M + 1) + j] -= mean[j];
        }
    }

    /* Calculate the m * m covariance matrix. */
#pragma omp parallel for private(j1, j2, i)
    for (j1 = 1; j1 < (M + 1); j1++) {
        for (j2 = j1; j2 < (M + 1); j2++) {
            symmat[j1 * (M + 1) + j2] = 0.0;
            for (i = 1; i < N + 1; i++) {
                symmat[j1 * (M + 1) + j2] +=
                    data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
            }
            symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
        }
    }
}

void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromOmp) {
    int i, j, fail;
    fail = 0;

    for (i = 1; i < (M + 1); i++) {
        for (j = 1; j < (N + 1); j++) {
            if (percentDiff(symmat[i * (N + 1) + j],
                            symmat_outputFromOmp[i * (N + 1) + j]) >
                PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }
    printf(
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: "
        "%d\n",
        PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main() {
    fprintf(stdout, BENCHMARK_INFO_STR, BENCHMARK_NAME, N);
    double t_start, t_end;

    DATA_TYPE* data;
    DATA_TYPE* symmat;
    DATA_TYPE* mean;
    DATA_TYPE* symmat_outputFromOmp;

    data = (DATA_TYPE*)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
    symmat = (DATA_TYPE*)malloc((M + 1) * (M + 1) * sizeof(DATA_TYPE));
    mean = (DATA_TYPE*)malloc((M + 1) * sizeof(DATA_TYPE));
    symmat_outputFromOmp =
        (DATA_TYPE*)malloc((M + 1) * (M + 1) * sizeof(DATA_TYPE));

    init_arrays(data);
    t_start = rtclock();
    covariance_omp(data, symmat_outputFromOmp, mean);
    t_end = rtclock();
    fprintf(stdout, "OMP Runtime: %0.6lfs\n", t_end - t_start);

    init_arrays(data);
    t_start = rtclock();
    covariance(data, symmat, mean);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

    compareResults(symmat, symmat_outputFromOmp);

    free(data);
    free(symmat);
    free(mean);
    free(symmat_outputFromOmp);

    return 0;
}
