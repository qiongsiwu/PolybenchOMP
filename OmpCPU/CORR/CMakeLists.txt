
set(SRC_DIR ./)

set(SRC_FILES
	correlation.c
)

add_executable(CORR ${SRC_FILES})

install(TARGETS CORR DESTINATION OMPCPU)
