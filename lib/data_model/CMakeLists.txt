set(LIB_NAME data_model)
project(${LIB_NAME})

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
)
set(SRC ${PROJECT_SOURCE_DIR}/src/data.cc
    ${PROJECT_SOURCE_DIR}/src/common.cc
)
set(INC ${PROJECT_SOURCE_DIR}/inc/data.hh
    ${PROJECT_SOURCE_DIR}/inc/common.hh
)

add_library(${LIB_NAME} ${SRC} ${INC})

if(DEFINED DEPENDSON)
    target_link_libraries(${LIB_NAME} ${DEPENDSON})
endif()