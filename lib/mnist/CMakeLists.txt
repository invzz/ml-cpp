
set(LIB_NAME mnist_handler)

project(${LIB_NAME})

IF (WIN32)
  # set stuff for windows
  find_package(pthread REQUIRED)
  set(DEPENDSON PThreads4W::PThreads4W)
ELSE()
ENDIF()

MESSAGE(STATUS "data: ${LIB_DIR}/data_model/inc")

include_directories(
  BEFORE
    ${LIB_DIR}/data_model/inc
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
)
set(DEPENDSON ${DEPENDSON} data_model)
set(SRC ${PROJECT_SOURCE_DIR}/src/${LIB_NAME}.cc)
set(INC ${PROJECT_SOURCE_DIR}/inc/${LIB_NAME}.hh)

add_library(${LIB_NAME} ${SRC} ${INC})

if(DEFINED DEPENDSON)
    target_link_libraries(${LIB_NAME} ${DEPENDSON}) 
endif()