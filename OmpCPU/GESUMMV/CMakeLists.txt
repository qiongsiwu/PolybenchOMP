
set(SRC_DIR ./)

set(SRC_FILES
	gesummv.c
)

add_executable(GESUMMV ${SRC_FILES})

install(TARGETS GESUMMV DESTINATION OMPCPU)
