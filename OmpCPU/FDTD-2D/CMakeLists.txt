
set(SRC_DIR ./)

set(SRC_FILES
	fdtd2d.c
)

add_executable(FDTD-2D ${SRC_FILES})

install(TARGETS FDTD-2D DESTINATION OMPCPU)
