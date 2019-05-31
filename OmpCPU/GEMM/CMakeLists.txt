
set(SRC_DIR ./)

set(SRC_FILES
	gemm.c
)

add_executable(GEMM ${SRC_FILES})

install(TARGETS GEMM DESTINATION OMPCPU)
