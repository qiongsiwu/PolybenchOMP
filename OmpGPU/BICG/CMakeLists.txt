
set(SRC_DIR ./)

set(SRC_FILES
	bicg.c
)

add_executable(BICG_T ${SRC_FILES})

install(TARGETS BICG_T DESTINATION OMPGPU)
