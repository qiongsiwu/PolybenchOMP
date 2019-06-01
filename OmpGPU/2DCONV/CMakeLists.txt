set(SRC_DIR ./)

set(SRC_FILES
  2DConvolution.c
)

add_executable(2DCONV_T ${SRC_FILES})

install(TARGETS 2DCONV_T DESTINATION OMPGPU)
