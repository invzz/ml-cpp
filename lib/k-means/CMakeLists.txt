set(LIB_NAME k-means)
project(${LIB_NAME})
include_directories( BEFORE ${PROJECT_SOURCE_DIR}/src 
${PROJECT_SOURCE_DIR}/inc 
${LIB_DIR}/data_model/inc
${LIB_DIR}/mnist/inc
${LIB_DIR}//inc
)
set(DEPENDSON data_model )
set(SRC ${PROJECT_SOURCE_DIR}/src/${LIB_NAME}.cc)
set(INC ${PROJECT_SOURCE_DIR}/inc/${LIB_NAME}.hh)
add_library(${LIB_NAME} ${SRC} ${INC})
if(DEFINED DEPENDSON) 
    target_link_libraries(${LIB_NAME} ${DEPENDSON}) 
endif()