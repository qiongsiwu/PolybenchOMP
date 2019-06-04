
set(SRC_DIR ./)

set(SRC_FILES
	gesummv.c
)

add_executable(GESUMMV_T ${SRC_FILES})

install(TARGETS GESUMMV_T DESTINATION OMPGPU)
