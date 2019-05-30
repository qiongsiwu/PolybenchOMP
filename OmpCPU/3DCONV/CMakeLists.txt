
set(SRC_DIR ./)

set(SRC_FILES
	3DConvolution.c
)

add_executable(3DCONV ${SRC_FILES})

install(TARGETS 3DCONV DESTINATION OMPCPU)
