
set(SRC_DIR ./)

set(SRC_FILES
	covariance.c
)

add_executable(COVAR_T ${SRC_FILES})

install(TARGETS COVAR_T DESTINATION OMPGPU)
