
set(SRC_DIR ./)

set(SRC_FILES
	3mm.c
)

add_executable(3MM ${SRC_FILES})

install(TARGETS 3MM DESTINATION OMPCPU)
