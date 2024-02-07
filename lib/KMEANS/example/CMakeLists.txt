set(APP_NAME KMEANS-EXAMPLE)

project(${APP_NAME})


include_directories(
    BEFORE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
    ${LIB_DIR}/mnist/inc
    ${LIB_DIR}/knn/inc
    ${LIB_DIR}/kmeans/inc
    ${LIB_DIR}/data_model/inc
)


set(DEPENDSON ${DEPENDSON} data_model mnist_handler knn kmeans)
set(SRC ${PROJECT_SOURCE_DIR}/src/main.cc)
set(INC ${PROJECT_SOURCE_DIR}/inc/main.hh)
add_executable(${APP_NAME} ${SRC} ${INC})

if(DEFINED DEPENDSON)
    MESSAGE(STATUS "[ ${APP_NAME} ] resolbing dependencies...: ${DEPENDSON}")
    target_link_libraries(${APP_NAME} ${DEPENDSON})
endif()