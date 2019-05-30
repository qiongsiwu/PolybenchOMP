
set(SRC_DIR ./)

set(SRC_FILES
	covariance.c
)

add_executable(COVAR ${SRC_FILES})

install(TARGETS COVAR DESTINATION OMPCPU)
