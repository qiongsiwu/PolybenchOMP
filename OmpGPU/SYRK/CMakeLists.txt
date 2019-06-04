
set(SRC_DIR ./)

set(SRC_FILES
	syrk.c
)

add_executable(SYRK_T ${SRC_FILES})

install(TARGETS SYRK_T DESTINATION OMPGPU)
