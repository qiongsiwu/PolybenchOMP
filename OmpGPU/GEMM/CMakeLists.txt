
set(SRC_DIR ./)

set(SRC_FILES
	gemm.c
)

add_executable(GEMM_T ${SRC_FILES})

install(TARGETS GEMM_T DESTINATION OMPCPU)
