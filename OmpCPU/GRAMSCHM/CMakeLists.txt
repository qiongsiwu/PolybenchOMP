
set(SRC_DIR ./)

set(SRC_FILES
	gramschmidt.c
)

add_executable(GRAMSCHM ${SRC_FILES})

install(TARGETS GRAMSCHM DESTINATION OMPCPU)
