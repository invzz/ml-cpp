set(LIB_NAME kmeans)
project(${LIB_NAME})
find_package(OpenMP REQUIRED)
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
    ${LIB_DIR}/data_model/inc
    ${LIB_DIR}/mnist/inc
    ${LIB_DIR}/inc
)
set(DEPENDSON data_model)
set(SRC ${PROJECT_SOURCE_DIR}/src/${LIB_NAME}.cc)
set(INC ${PROJECT_SOURCE_DIR}/inc/${LIB_NAME}.hh)
add_library(${LIB_NAME} ${SRC} ${INC})
add_subdirectory(${LIB_DIR}/KMEANS/example)
target_compile_options(${LIB_NAME} PRIVATE -Wall ${OpenMP_CXX_FLAGS})

if(DEFINED DEPENDSON)
    MESSAGE(STATUS "[ ${LIB_NAME} ] resolving dependencies...: ${DEPENDSON}")
    target_link_libraries(${LIB_NAME} PUBLIC ${DEPENDSON} ${OpenMP_CXX_FLAGS})
endif()