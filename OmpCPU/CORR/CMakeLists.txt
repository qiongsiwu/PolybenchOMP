
set(SRC_DIR ./)

set(SRC_FILES
	correlation.c
)

add_executable(CORR ${SRC_FILES})

target_link_libraries(CORR m)

install(TARGETS CORR DESTINATION OMPCPU)
