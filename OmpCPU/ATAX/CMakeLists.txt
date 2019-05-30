
set(SRC_DIR ./)

set(SRC_FILES
	atax.c
)

add_executable(ATAX ${SRC_FILES})

install(TARGETS ATAX DESTINATION OMPCPU)
