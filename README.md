# PolybenchOMP
OpenMP version of PolybenchGPU

The algorithms in [PolybenchGPU](http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/GPU/) 
are reimplemented using OpenMP. There are existing implementations, such as 
[Unibench](https://github.com/UofT-EcoSystem/Unibench), but Unibench suffers from incorrect implementations that may cause
incorrect computation results or invalid nvprof results. This reimplementaiton is aimed at solving these issues. 

### Prerequisite ###
 - LLVM with OpenMP. See how you can get both [here](https://llvm.org/docs/GettingStarted.html) and follow my blog 
 [here](https://qiongsiwu.github.io/llvm/openmp/2019/05/29/how-to-build-openmp-offloading-lib.html) 
 for more details on how to build llvm and OpenMP target offloading. 

### Building and Running the Benchmarks ###
To quickly start building and running, clone this repository first and follow the steps below. 
1. Clone this git repository. 
2. Go to the directory `PolybenchOMP` and create a build directory. 
```bash
cd PolybencOMP
mkdir build
```
2. Configure the project. 
```bash
# Set the path to LLVM installation
export LLVM_INSTALL=/where/llvm/is/installed

# Go to the build directory
cd build

# Configure the project
cmake -DCMAKE_BUILD_TYPE=Release -DRUN_TEST=1 \
      -DCMAKE_C_COMPILER=$LLVM_INSTALL/bin/clang \
      -DCMAKE_CXX_COMPILER=$LLVM_INSTALL/bin/clang++ \
      -DOPENMP_LIBRARIES=$LLVM_INSTALL/lib \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
      -DOPENMP_INCLUDES=~/llvm-project/install/include \
      ..
```
3. Install all the benchmarks. 
```bash 
make install -j16
```

### CMake Options ###
 - `RUN_OFFLOAD_GPU` default is on. This option turns on building for nvptx target offloading. 
 - `RUN_TEST` default is off. When turned on, the benchmarks compare results with a servial version of the algorithm. For now this option is only controlling OMPGPU targets. OMPCPU always have the test code on. 
 
### Known Limitations ###
 - Running on Windows is neither tested nor supported. 
 - Target offloading is not supported on macOS. 
 
