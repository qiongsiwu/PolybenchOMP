
set(SRC_DIR ./)

set(SRC_FILES
	syr2k.c
)

add_executable(SYR2K_T ${SRC_FILES})

install(TARGETS SYR2K_T DESTINATION OMPGPU)
