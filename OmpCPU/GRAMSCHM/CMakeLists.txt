
set(SRC_DIR ./)

set(SRC_FILES
	gramschmidt.c
)

add_executable(GRAMSCHM ${SRC_FILES})

target_link_libraries(GRAMSCHM m)

install(TARGETS GRAMSCHM DESTINATION OMPCPU)
