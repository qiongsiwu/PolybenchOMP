
set(SRC_DIR ./)

set(SRC_FILES
	syrk.c
)

add_executable(SYRK ${SRC_FILES})

install(TARGETS SYRK DESTINATION OMPCPU)
