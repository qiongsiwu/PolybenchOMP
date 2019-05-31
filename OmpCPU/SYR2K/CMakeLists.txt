
set(SRC_DIR ./)

set(SRC_FILES
	syr2k.c
)

add_executable(SYR2K ${SRC_FILES})

install(TARGETS SYR2K DESTINATION OMPCPU)
