
set(SRC_DIR ./)

set(SRC_FILES
	atax.c
)

add_executable(ATAX_T ${SRC_FILES})

install(TARGETS ATAX_T DESTINATION OMPGPU)
