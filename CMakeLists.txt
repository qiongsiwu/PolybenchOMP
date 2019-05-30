cmake_minimum_required(VERSION 3.1)
project(PolybenchOMP)

# Options for the benchmark size
option(RUN_TEST "Compare with CPU execution" OFF)
if(RUN_TEST)
    message("Run test to compare results with CPU execution")
    add_definitions(-DRUN_TEST=1)
endif()

option(RUN_BENCHMARK "Use large datasets" OFF)
if(RUN_BENCHMARK)
    add_definitions(-DRUN_BENCHMARK=1)
endif()

option(RUN_POLYBENCH_SIZE "Run with problem sizes in original polybench" OFF)
if(RUN_POLYBENCH_SIZE)
	add_definitions(-DRUN_POLYBENCH_SIZE=1)
endif()

# Use C99 standard
set (CMAKE_C_STANDARD 99)

OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" ON)
# Find OpenMP
if(APPLE AND USE_OpenMP)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()

if(USE_OpenMP)
    find_package(OpenMP REQUIRED)
endif(USE_OpenMP)

if (OPENMP_FOUND)
    include_directories("${OPENMP_INCLUDES}")
    link_directories("${OPENMP_LIBRARIES}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

add_subdirectory(OmpCPU)
