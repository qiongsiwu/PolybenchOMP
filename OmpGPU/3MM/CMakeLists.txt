
set(SRC_DIR ./)

set(SRC_FILES
	3mm.c
)

add_executable(3MM_T ${SRC_FILES})

install(TARGETS 3MM_T DESTINATION OMPCPU)
