set(SRC_DIR ./)

set(SRC_FILES
  2DConvolution.c
)

add_executable(2DCONV ${SRC_FILES})

install(TARGETS 2DCONV DESTINATION OMPCPU)
