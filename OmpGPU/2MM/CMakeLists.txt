
set(SRC_DIR ./)

set(SRC_FILES
  2mm.c
)

add_executable(2MM_T ${SRC_FILES})

install(TARGETS 2MM_T DESTINATION OMPGPU)
