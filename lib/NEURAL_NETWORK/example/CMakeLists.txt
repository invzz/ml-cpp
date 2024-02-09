set(APP_NAME NEURAL-NETWORK-EXAMPLE)

project(${APP_NAME})

include_directories(
    BEFORE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
    ${LIB_DIR}/iris/inc
    ${LIB_DIR}/NEURAL_NETWORK/inc
    ${LIB_DIR}/data_model/inc
)

set(DEPENDSON ${DEPENDSON} data_model iris_handler neural_network)
set(SRC ${PROJECT_SOURCE_DIR}/src/main.cc)
set(INC ${PROJECT_SOURCE_DIR}/inc/main.hh)
add_executable(${APP_NAME} ${SRC} ${INC})

if(DEFINED DEPENDSON)
    MESSAGE(STATUS "[ ${APP_NAME} ]  dependencies...: ${DEPENDSON}")
    target_link_libraries(${APP_NAME} ${DEPENDSON})
endif()