
set(SRC_DIR ./)

set(SRC_FILES
	correlation.c
)

add_executable(CORR_T ${SRC_FILES})

target_link_libraries(CORR_T m)

install(TARGETS CORR_T DESTINATION OMPGPU)
