set(APP_NAME etl)

project(${APP_NAME})




include_directories(
    BEFORE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
    ${LIB_DIR}/mnist/inc
    ${LIB_DIR}/knn/inc
    ${LIB_DIR}/data_model/inc
)

set(DEPENDSON ${DEPENDSON} mnist_handler knn data_model)
set(SRC ${PROJECT_SOURCE_DIR}/src/main.cc )
set(INC ${PROJECT_SOURCE_DIR}/inc/main.hh )
add_executable(${APP_NAME} ${SRC} ${INC} )
if(DEFINED DEPENDSON)
    MESSAGE(STATUS "[ ${APP_NAME} ] resolbing dependencies...: ${DEPENDSON}")
    target_link_libraries(${APP_NAME} ${DEPENDSON}) 
endif()