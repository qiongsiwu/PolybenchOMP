
set(SRC_DIR ./)

set(SRC_FILES
	mvt.c
)

add_executable(MVT_T ${SRC_FILES})

install(TARGETS MVT_T DESTINATION OMPGPU)
