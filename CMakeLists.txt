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

# OpenMP Options
option(OMP_DISABLE "Disable OpenMP" OFF)
if(NOT OMP_DISABLE)
   	find_package(OpenMP)
   	if(OPENMP_FOUND)
   		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   	endif()
endif()

add_subdirectory(OmpCPU)
