
set(SRC_DIR ./)

set(SRC_FILES
	mvt.c
)

add_executable(MVT ${SRC_FILES})

install(TARGETS MVT DESTINATION OMPCPU)
