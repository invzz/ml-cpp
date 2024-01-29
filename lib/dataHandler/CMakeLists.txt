set(LIB_NAME dataHandler)

project(${LIB_NAME})

IF (WIN32)
  # set stuff for windows
  find_package(pthread REQUIRED)
  set(DEPENDSON PThreads4W::PThreads4W)
ELSE()
ENDIF()

include_directories(
    BEFORE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
)
aux_source_directory(SRC ${PROJECT_SOURCE_DIR}/src/ )
aux_source_directory(INC ${PROJECT_SOURCE_DIR}/inc )

set(SRC ${PROJECT_SOURCE_DIR}/src/data_handler.cc
        ${PROJECT_SOURCE_DIR}/src/data.cc
)
set(INC ${PROJECT_SOURCE_DIR}/inc/data_handler.hh
        ${PROJECT_SOURCE_DIR}/inc/data.hh
)

# MESSAGE(STATUS "${LIB_NAME} SRC...: ${SRC}")
# MESSAGE(STATUS "${LIB_NAME} INC...: ${INC}")

add_library(${LIB_NAME} ${SRC} ${INC})

if(DEFINED DEPENDSON)
    target_link_libraries(${LIB_NAME} ${DEPENDSON}) 
endif()