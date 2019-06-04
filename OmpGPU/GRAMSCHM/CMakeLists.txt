
set(SRC_DIR ./)

set(SRC_FILES
	gramschmidt.c
)

add_executable(GRAMSCHM_T ${SRC_FILES})

target_link_libraries(GRAMSCHM_T m)

install(TARGETS GRAMSCHM_T DESTINATION OMPGPU)
