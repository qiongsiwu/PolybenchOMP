
set(SRC_DIR ./)

set(SRC_FILES
	bicg.c
)

add_executable(BICG ${SRC_FILES})

install(TARGETS BICG DESTINATION OMPCPU)
