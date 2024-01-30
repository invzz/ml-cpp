set(LIB_NAME tqueue)

project(${LIB_NAME})


include_directories(
    BEFORE
    ${PROJECT_SOURCE_DIR}/inc
)

#set(DEPENDSON ${DEPENDSON} mnist_handler knn data_model)
set(INC ${PROJECT_SOURCE_DIR}/inc/${LIB_NAME}.hh )
add_library(${LIB_NAME} INTERFACE ${INC} )

if(DEFINED DEPENDSON)
    MESSAGE(STATUS "[ ${LIB_NAME} ] resolving dependencies...: ${DEPENDSON}")
    target_link_libraries(${LIB_NAME} ${DEPENDSON}) 
endif()